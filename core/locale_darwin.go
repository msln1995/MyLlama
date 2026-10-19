//go:build darwin

package core

import (
	"os/exec"
	"strings"
)

// readSystemLocale reads the macOS system locale via `defaults read`:
// returns the first language from AppleLanguages (e.g. "zh-Hans-CN" / "en-US"),
// parsed by localeToLanguage via prefix match. Returns empty string on failure;
// detectSystemLanguage falls back to "en".
func readSystemLocale() string {
	out, err := exec.Command("defaults", "read", "-g", "AppleLanguages").Output()
	if err != nil {
		return ""
	}
	// Output format: ( "zh-Hans-CN", "en-CN" ) — take the first quoted string.
	s := string(out)
	start := strings.Index(s, `"`)
	if start < 0 {
		return ""
	}
	end := strings.Index(s[start+1:], `"`)
	if end < 0 {
		return ""
	}
	return s[start+1 : start+1+end]
}
