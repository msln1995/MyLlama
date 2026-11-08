// sd engine detection and probing for the media domain. The engine is a
// standalone sd-cli binary (stable-diffusion.cpp); P1 supports manual setup
// (the user places the binary in the managed media-runtime dir or configures
// an engine dir via settings) — the automated installer lands in P2. Version
// probing shells out once and caches, same discipline as the llama.cpp probe
// in llamacpp.go.
package core

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// mediaEngineBinaryName is the executable name per OS.
func mediaEngineBinaryName() string {
	if runtime.GOOS == "windows" {
		return "sd-cli.exe"
	}
	return "sd-cli"
}

// MediaRuntimeStatus describes the detected sd engine. Backend is reported
// only when it can be stated honestly: "unknown" until a real probe says
// otherwise (plan §4.2: unknown is never treated as zero or "enough").
type MediaRuntimeStatus struct {
	Installed  bool   `json:"installed"`
	EngineDir  string `json:"engineDir"`
	EnginePath string `json:"enginePath"`
	Version    string `json:"version"`
	Commit     string `json:"commit"`
	Backend    string `json:"backend"` // best-effort: "unknown" when not detectable from --version
	Error      string `json:"error,omitempty"`
}

// mediaRuntimeProbeOnce / mediaRuntimeProbed cache the probe result per
// process; tests reset them via mediaResetRuntimeCache.
var (
	mediaRuntimeMu       sync.Mutex
	mediaRuntimeProbed   bool
	mediaRuntimeCached   MediaRuntimeStatus
	mediaProbeVersionCmd = func(enginePath string) (string, error) {
		cmd := exec.Command(enginePath, "--version")
		hideWindow(cmd)
		out, err := cmd.CombinedOutput()
		return string(out), err
	}
)

// MediaRuntimeStatusNow resolves and probes the engine (cached per process;
// the cache is dropped when the engine dir setting changes).
func MediaRuntimeStatusNow() MediaRuntimeStatus {
	mediaRuntimeMu.Lock()
	defer mediaRuntimeMu.Unlock()
	dir := activeMediaRuntimeDir()
	enginePath := filepath.Join(dir, mediaEngineBinaryName())
	if mediaRuntimeProbed && mediaRuntimeCached.EngineDir == dir {
		return mediaRuntimeCached
	}
	st := MediaRuntimeStatus{EngineDir: dir, EnginePath: enginePath, Backend: "unknown"}
	fi, err := os.Stat(enginePath)
	switch {
	case err != nil:
		st.Error = "sd engine not found (install or configure the engine directory)"
	case fi.IsDir():
		st.Error = "engine path is a directory, expected the sd-cli binary"
	default:
		st.Installed = true
		version, err := mediaProbeVersionCmd(enginePath)
		if err != nil {
			st.Error = "engine version probe failed: " + err.Error()
			st.Installed = true // binary exists; the probe failure is reported verbatim
		} else {
			st.Version, st.Commit = parseMediaEngineVersion(version)
		}
	}
	mediaRuntimeProbed = true
	mediaRuntimeCached = st
	return st
}

// mediaResetRuntimeCache drops the cached probe (config change / tests).
func mediaResetRuntimeCache() {
	mediaRuntimeMu.Lock()
	mediaRuntimeProbed = false
	mediaRuntimeMu.Unlock()
}

// mediaEngineVersionRe extracts "stable-diffusion.cpp version X, commit Y".
var mediaEngineVersionRe = regexp.MustCompile(`version\s+(\S+),\s*commit\s+([0-9a-f]+)`)

// parseMediaEngineVersion pulls the version and commit out of sd-cli
// --version output ("stable-diffusion.cpp version unknown, commit 5f8c19f").
// Unparseable output yields empty fields with no error — the caller reports
// what it knows instead of inventing values.
func parseMediaEngineVersion(out string) (version, commit string) {
	if m := mediaEngineVersionRe.FindStringSubmatch(strings.TrimSpace(out)); m != nil {
		return m[1], m[2]
	}
	return "", ""
}

// mediaValidateEngineForProfile checks the engine snapshot against a
// variant's profile. P1 has no engine-version compatibility database yet:
// any detected engine passes, and the honest "unknown" fields are surfaced
// to the UI rather than guessed (plan §4.2 安装完整性: 组件齐备但引擎不兼容
// 仍不能开始 — the gate here is presence + runnable probe; a deep compat
// matrix is a P2 concern with the installer).
func mediaValidateEngineForProfile(st MediaRuntimeStatus) error {
	if !st.Installed {
		return fmt.Errorf("sd engine not installed")
	}
	if st.Error != "" {
		return fmt.Errorf("sd engine probe failed: %s", st.Error)
	}
	return nil
}

// intPtr formats an int for argv without allocation surprises.
func mediaArgInt(v int) string { return strconv.Itoa(v) }
