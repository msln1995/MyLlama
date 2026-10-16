#!/usr/bin/env bash
# MyLlama macOS packaging (fork-only tooling; runs on a Mac).
#
# What this does, in order:
#   1. Compiles the Go/Wails app with -tags production for darwin, once per
#      requested architecture (arm64 and x86_64 by default). Both slices are
#      cross-compiled from the same machine using the installed macOS SDK, so
#      "Intel vs Apple Silicon" costs one extra `go build`, not extra code:
#      the llama.cpp runtime download is already architecture-aware
#      (core/llamacpp.go maps darwin -> the "macos" asset family and matches
#      the x64 / arm64 tags), and acceleration is resolved per arch there too
#      (arm64 -> Metal, x64 -> CPU-only; Intel Macs have no Metal backend in
#      llama.cpp's release builds).
#   2. `lipo`s the slices into a universal executable (MACOS_ARCH=arm64 or
#      x86_64 skips this and ships a thin bundle instead).
#   3. Assembles a real .app bundle: Contents/MacOS/<bin>, Info.plist,
#      PkgInfo, and an AppIcon.icns generated from build/appicon.png. The bare
#      binary already runs (that is how the llama.cpp chain gets exercised in
#      CI-adjacent local testing), but a bundle is what gives the Dock icon,
#      the product name and a sane Gatekeeper presentation.
#   4. Signs the bundle ad-hoc (`codesign -s -`). There is no Developer ID
#      identity here and a fork cannot use upstream's certificate, so the
#      artifact stays un notarized: see the README's macOS install notes for the
#      quarantine workaround.
#   5. Produces the two distributables (zip via ditto, dmg via hdiutil) plus
#      their SHA-256 digests, under build/bin/.
#
# Deliberately NOT part of .github/workflows/ci.yml: macOS runners are billed
# even for public repositories (Linux/Windows minutes are the free ones), so
# this is a local-first release path. Upstream can decide separately whether a
# build-macos CI job is worth the minutes.
#
# Outputs: build/bin/MyLlama.app (left for local testing) plus the two
# distributables and their .sha256 files.
#
# Usage (from the repo root, after `cd frontend && npm ci && npm run build`):
#   scripts/macos_package.sh
#   MACOS_ARCH=arm64 scripts/macos_package.sh          # thin arm64 bundle only
#   FORK_TAG=v0.5.3-fork.2 scripts/macos_package.sh    # artifact naming
#   BUNDLE_ID=com.example.myllama scripts/macos_package.sh
#
# Env knobs and their defaults:
#   MACOS_ARCH   universal | arm64 | amd64   (default: universal)
#   FORK_TAG     release tag used in artifact names (default: <core/VERSION>-fork.1)
#   BUNDLE_ID    CFBundleIdentifier          (default: com.msln1995.myllama,
#                mirroring this fork's Android applicationId)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="MyLlama"
MACOS_ARCH="${MACOS_ARCH:-universal}"
BUNDLE_ID="${BUNDLE_ID:-com.msln1995.myllama}"
OUT_DIR="$ROOT/build/bin"
WORK_DIR="$(mktemp -d -t myllama-macos-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

# --- preflight -------------------------------------------------------------

[ "$(uname -s)" = "Darwin" ] || { echo "this script must run on macOS" >&2; exit 1; }
command -v go >/dev/null || { echo "go not found in PATH" >&2; exit 1; }
command -v xcrun >/dev/null || { echo "Xcode command line tools not found" >&2; exit 1; }

# CGO needs a usable Apple toolchain. When `xcode-select` points at a full Xcode
# whose license has not been agreed, every `xcrun` call fails and the cgo build
# dies; the standalone Command Line Tools SDK is a valid substitute and selecting
# it via DEVELOPER_DIR needs no sudo and no license acceptance. Only fall back to
# it when the selected directory is actually unusable, so a machine with a
# properly licensed Xcode keeps using it (and its matching SDK).
if ! xcrun --show-sdk-path >/dev/null 2>&1 \
   && [ -d /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk ]; then
  export DEVELOPER_DIR=/Library/Developer/CommandLineTools
  echo "==> selected Xcode unusable (license?); building against Command Line Tools"
fi

# frontend/dist is a compile-time go:embed input: a missing or placeholder dist
# yields an app that launches to a blank window, so refuse instead of warn.
[ -f frontend/dist/index.html ] && [ -d frontend/dist/assets ] || {
  echo "frontend/dist looks incomplete; run: cd frontend && npm ci && npm run build" >&2
  exit 1
}

BASE_VERSION="$(tr -d 'v\n' < core/VERSION)"
FORK_TAG="${FORK_TAG:-v${BASE_VERSION}-fork.1}"
SHORT_VERSION="${FORK_TAG#v}"          # e.g. 0.5.3-fork.1
BUILD_VERSION="$(echo "$SHORT_VERSION" | tr '-' '.')"  # CFBundleVersion: 0.5.3.fork.1

# GOARCH names are used throughout (Go spells the Intel slice "amd64", while
# Mach-O / `lipo -archs` call it "x86_64"); "x86_64" is accepted as an alias so
# either spelling works at the command line.
case "$MACOS_ARCH" in
  universal) SLICES=(arm64 amd64) ;;
  arm64)     SLICES=(arm64) ;;
  amd64|x86_64) SLICES=(amd64); MACOS_ARCH=amd64 ;;
  *) echo "MACOS_ARCH must be universal | arm64 | amd64 (got: $MACOS_ARCH)" >&2; exit 1 ;;
esac

echo "==> $APP_NAME $FORK_TAG ($MACOS_ARCH, id $BUNDLE_ID)"

# --- 1. compile one slice per architecture --------------------------------

# Same flag set as the CI Linux job (plain go build, no wails CLI, no -H
# windowsgui): go:embed pulls in frontend/dist, CGO is required for WKWebView.
# CGO_TARGET_HDRS are unnecessary because the macOS SDK ships both arch slices,
# so GOARCH alone is enough to cross-compile.
for arch in "${SLICES[@]}"; do
  echo "==> go build darwin/$arch"
  mkdir -p "$WORK_DIR/$arch"
  CGO_ENABLED=1 GOOS=darwin GOARCH="$arch" \
    go build -tags production -trimpath -buildvcs=false -ldflags="-w -s" \
      -o "$WORK_DIR/$arch/$APP_NAME" .
done

# --- 2. universal executable ----------------------------------------------

mkdir -p "$WORK_DIR/bundle/Contents/MacOS"
BIN_OUT="$WORK_DIR/bundle/Contents/MacOS/$APP_NAME"
SLICE_PATHS=()
for arch in "${SLICES[@]}"; do
  SLICE_PATHS+=("$WORK_DIR/$arch/$APP_NAME")
done
if [ "${#SLICE_PATHS[@]}" -gt 1 ]; then
  # One Mach-O carrying both arch slices: the same .app opens on Apple Silicon
  # and Intel Macs, and macOS picks the native slice at exec time.
  lipo -create -output "$BIN_OUT" "${SLICE_PATHS[@]}"
else
  cp "${SLICE_PATHS[0]}" "$BIN_OUT"
fi
chmod +x "$BIN_OUT"
lipo -archs "$BIN_OUT" | sed 's/^/    archs: /'

# --- 3. bundle: Info.plist, PkgInfo, icon ---------------------------------

cat > "$WORK_DIR/bundle/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>$APP_NAME</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIdentifier</key>
	<string>$BUNDLE_ID</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>$APP_NAME</string>
	<key>CFBundleDisplayName</key>
	<string>$APP_NAME</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$SHORT_VERSION</string>
	<key>CFBundleVersion</key>
	<string>$BUILD_VERSION</string>
	<key>LSMinimumSystemVersion</key>
	<string>11.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSHumanReadableCopyright</key>
	<string>GPL-3.0 - upstream https://github.com/CodeNeow/MyLlama</string>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$WORK_DIR/bundle/Contents/PkgInfo"

# A malformed Info.plist silently degrades to LaunchServices defaults (wrong
# product name, no bundle identity), so lint it before it reaches the bundle.
plutil -lint "$WORK_DIR/bundle/Contents/Info.plist" >/dev/null

# .icns from the 512px source PNG (iconutil wants the whole iconset grid).
mkdir -p "$WORK_DIR/AppIcon.iconset"
ICONSRC="build/appicon.png"
for spec in "16:icon_16x16" "32:icon_16x16@2x" "32:icon_32x32" "64:icon_32x32@2x" \
            "128:icon_128x128" "256:icon_128x128@2x" "256:icon_256x256" "512:icon_256x256@2x" \
            "512:icon_512x512" "1024:icon_512x512@2x"; do
  px="${spec%%:*}"; name="${spec#*:}"
  sips -z "$px" "$px" -s format png "$ICONSRC" --out "$WORK_DIR/AppIcon.iconset/$name.png" >/dev/null
done
mkdir -p "$WORK_DIR/bundle/Contents/Resources"
iconutil -c icns "$WORK_DIR/AppIcon.iconset" -o "$WORK_DIR/bundle/Contents/Resources/AppIcon.icns"

# --- 4. ad-hoc signature --------------------------------------------------

mkdir -p "$OUT_DIR"
APP="$OUT_DIR/$APP_NAME.app"
rm -rf "$APP"
cp -R "$WORK_DIR/bundle" "$APP"
chmod +x "$APP/Contents/MacOS/$APP_NAME"
# Ad-hoc identity ("-" = no Developer ID). --force re-signs over the linker's
# automatic signature so the bundle is sealed as one unit.
codesign --force --sign - --timestamp=none "$APP" >/dev/null 2>&1
# --deep --strict is the check that passes for an ad-hoc bundle: a plain
# `codesign --verify` additionally evaluates the designated requirement, which
# an ad-hoc signature can never satisfy (no Developer ID anchor), so it would
# fail for a reason that has nothing to do with the bundle being broken.
codesign --verify --deep --strict "$APP"
echo "    codesign: bundle seals cleanly ($(codesign -dvv "$APP" 2>&1 | sed -n 's/^Signature=//p' | head -1) identity, no Developer ID)"
for key in CFBundleIdentifier CFBundleShortVersionString CFBundleVersion; do
  echo "    $key = $(plutil -extract "$key" raw "$APP/Contents/Info.plist")"
done
# Distribution caveat this script cannot test locally: an ad-hoc signature
# cannot be notarized, so a browser-downloaded copy carries
# com.apple.quarantine and Gatekeeper blocks the first launch. `spctl -a` only
# proves that when local assessments are enabled (check `spctl --status`); on a
# machine with them disabled it reports "accepted" for everything, so the
# README documents the right-click-open / `xattr -dr com.apple.quarantine`
# workaround rather than relying on a check here.

# --- 5. distributables ----------------------------------------------------

NAME="$APP_NAME-${FORK_TAG}-macos-${MACOS_ARCH}"
ZIP="$OUT_DIR/$NAME.zip"
DMG="$OUT_DIR/$NAME.dmg"
[ -f "$ZIP" ] && rm -f "$ZIP"
[ -f "$DMG" ] && rm -f "$DMG"

# ditto preserves the bundle metadata and the code signature (plain zip does not).
ditto -c -k --keepParent "$APP" "$ZIP"

# dmg: the app plus an /Applications alias so the drop target is obvious.
LNK_DIR="$WORK_DIR/dmg"
mkdir -p "$LNK_DIR"
cp -R "$APP" "$LNK_DIR/"
ln -s /Applications "$LNK_DIR/Applications"
hdiutil create -volname "$APP_NAME $FORK_TAG" -srcfolder "$LNK_DIR" -ov -format UDZO "$DMG" >/dev/null

for f in "$ZIP" "$DMG"; do
  shasum -a 256 "$f" | tee "$f.sha256" | sed 's/^/    sha256: /'
done

echo "==> done"
echo "    bundle: $APP"
ls -la "$OUT_DIR" | grep "$NAME" | sed 's/^/    /'
