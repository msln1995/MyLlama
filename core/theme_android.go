//go:build android

package core

// getSystemTheme returns the Android system theme fallback for Go backend.
// In the Android app sandbox, os/exec cannot spawn child processes,
// so system theme is natively detected and dynamically updated on the
// frontend via WebView matchMedia and MainActivity onConfigurationChanged.
func getSystemTheme() string {
	return "light"
}
