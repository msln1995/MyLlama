//go:build android

package core

import (
	"os/exec"
	"strings"
)

// readSystemLocale reads the Android system locale via `getprop`:
// returns persist.sys.locale (e.g. "zh-CN" / "en-US"), parsed by localeToLanguage
// via prefix match. Returns empty string on failure; detectSystemLanguage falls
// back to "en".
func readSystemLocale() string {
	out, err := exec.Command("getprop", "persist.sys.locale").Output()
	if err != nil || len(out) == 0 {
		return ""
	}
	return strings.TrimSpace(string(out))
}
