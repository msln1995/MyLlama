//go:build android

package core

import (
	"os/exec"
	"strings"
)

// getSystemTheme returns the Android system theme: "dark" if ui_night_mode is
// 1 or 2, else "light". Returns "light" on failure.
func getSystemTheme() string {
	out, err := exec.Command("cmd", "uimode", "get").Output()
	if err != nil {
		return "light"
	}
	s := strings.TrimSpace(string(out))
	if s == "night" || s == "1" || s == "2" {
		return "dark"
	}
	return "light"
}
