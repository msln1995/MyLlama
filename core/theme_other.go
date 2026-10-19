//go:build !darwin && !android

package core

// getSystemTheme returns the system theme on platforms without a detection
// mechanism: always "light". Windows and Linux desktop can be extended later
// with platform-specific APIs if needed.
func getSystemTheme() string {
	return "light"
}
