// Media-generation path derivation. The media domain owns three directories
// and two state files, all mirroring the chat-model layout rules:
//   - the sd engine lives in a managed media-runtime dir resolved like the
//     llama.cpp runtime (internal storage on Android, where exec is allowed —
//     see paths.go);
//   - model weights and generated outputs land in large-file dirs resolved
//     like the chat models dir (app-specific external storage on Android);
//   - job/works state files resolve through resolveStateFile like every
//     other state file.
//
// The dirs are function-valued vars (same injection style as defaultModelsDir
// / defaultLlamaCppDir) so tests can pin locations. The *active* dirs may be
// overridden by config (mediaSettings below); the override is applied by the
// mediaSettings appliers in config.go.
package core

import (
	"log"
	"os"
	"path/filepath"
	"sync"
)

const (
	mediaRuntimeDirName = "media-runtime"
	mediaModelsDirName  = "Media-Models"
	mediaOutputsDirName = "Media-Outputs"

	mediaJobsFileName     = "myllama-media-jobs.json"
	mediaWorksFileName    = "myllama-media-works.json"
	mediaInstallsFileName = "myllama-media-installs.json"
)

// defaultMediaRuntimeDir resolves the managed sd engine directory: the same
// location rules as the llama.cpp runtime (resolveStateFile(mediaRuntimeDirName))
// so the exec-ability constraint is satisfied on every platform.
var defaultMediaRuntimeDir = func() string {
	return resolveStateFile(mediaRuntimeDirName)
}

// defaultMediaModelsDir resolves the default media weights directory: bare
// cwd-relative on Windows, under the app-data base on other desktops, and on
// Android under the app-specific external-storage base (multi-GB weights need
// the shared-disk headroom; same shape as defaultModelsDir in gguf.go).
var defaultMediaModelsDir = func() string {
	if pathsGOOS == "android" {
		if base := pathsAndroidModelsBase(); base != "" {
			return filepath.Join(base, mediaModelsDirName)
		}
	}
	return resolveStateFile(mediaModelsDirName)
}

// defaultMediaOutputsDir resolves the default generated-outputs directory:
// same placement rules as the media weights dir (outputs are media files too
// and on Android belong on the shared-disk base so users can reach them).
var defaultMediaOutputsDir = func() string {
	if pathsGOOS == "android" {
		if base := pathsAndroidModelsBase(); base != "" {
			return filepath.Join(base, mediaOutputsDirName)
		}
	}
	return resolveStateFile(mediaOutputsDirName)
}

// Guarded overrides (config-applied). Empty values mean "use the defaults
// above"; the appliers in config.go write them under mediaSettingsMu.
var (
	mediaSettingsMu       sync.Mutex
	mediaEngineDirSetting string
	mediaModelsDirSetting string
	mediaOutputsSetting   string
)

// activeMediaRuntimeDir returns the sd engine directory in effect: the
// configured override when set, otherwise the managed default.
func activeMediaRuntimeDir() string {
	mediaSettingsMu.Lock()
	dir := mediaEngineDirSetting
	mediaSettingsMu.Unlock()
	if dir != "" {
		return dir
	}
	return defaultMediaRuntimeDir()
}

// activeMediaModelsDir returns the media weights directory in effect.
func activeMediaModelsDir() string {
	mediaSettingsMu.Lock()
	dir := mediaModelsDirSetting
	mediaSettingsMu.Unlock()
	if dir != "" {
		return dir
	}
	return defaultMediaModelsDir()
}

// activeMediaOutputsDir returns the generated-outputs directory in effect.
func activeMediaOutputsDir() string {
	mediaSettingsMu.Lock()
	dir := mediaOutputsSetting
	mediaSettingsMu.Unlock()
	if dir != "" {
		return dir
	}
	return defaultMediaOutputsDir()
}

// mediaJobsFilePath resolves the media jobs state file path.
func mediaJobsFilePath() string {
	return resolveStateFile(mediaJobsFileName)
}

// mediaInstallsFilePath resolves the media installs state file path.
func mediaInstallsFilePath() string {
	return resolveStateFile(mediaInstallsFileName)
}

// mediaWorksFilePath resolves the media works state file path.
func mediaWorksFilePath() string {
	return resolveStateFile(mediaWorksFileName)
}

// ensureMediaDirs creates the three media directories on demand (best-effort;
// callers surface their own errors when a needed dir is still missing).
func ensureMediaDirs() {
	for _, dir := range []string{activeMediaRuntimeDir(), activeMediaModelsDir(), activeMediaOutputsDir()} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Printf("[WARN] Cannot create media directory %s: %v", dir, err)
		}
	}
}
