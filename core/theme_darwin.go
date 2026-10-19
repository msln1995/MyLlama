//go:build darwin

package core

import (
	"os/exec"
	"strings"
)

// getSystemTheme returns the macOS system appearance: "dark" if AppleInterfaceStyle
// is "Dark", else "light". Returns "light" on failure or when the key is absent.
func getSystemTheme() string {
	out, err := exec.Command("defaults", "read", "-g", "AppleInterfaceStyle").Output()
	if err != nil {
		return "light"
	}
	if strings.TrimSpace(string(out)) == "Dark" {
		return "dark"
	}
	return "light"
}
