//go:build android

package core

import (
	"strings"
)

// readSystemLocale reads the Android system locale.
// In the Android app sandbox, os/exec cannot spawn child processes (SELinux denies
// `getprop` execution by untrusted_app). We read system build properties in-process
// via readProcFile without spawning child processes.
func readSystemLocale() string {
	for _, propFile := range []string{"/system/build.prop", "/vendor/build.prop", "/product/build.prop"} {
		content := readProcFile(propFile)
		if content == "" {
			continue
		}
		for _, line := range strings.Split(content, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "ro.product.locale=") {
				return strings.TrimPrefix(line, "ro.product.locale=")
			}
		}
	}
	return ""
}
