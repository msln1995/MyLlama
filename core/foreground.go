package core

// ─── Android foreground service (keep the server running) ─────────
//
// A backgrounded Android app is not merely deprioritised: the OS moves it to
// the little-core cpuset and can freeze the process outright. Measured on the
// vivo X60 (Exynos 1080, Android 13) with the app sitting in the background:
// Cpus_allowed_list 0-3 (the four A55 cores at 2.002 GHz; /dev/cpuset/
// background/cpus = 0-3, top-app and foreground both allow 0-7), cgroup
// freezer:/frozen and task state D. llama-server is a child of the app
// process and inherits those cgroups, so switching apps or turning the screen
// off stops an in-flight generation mid-token.
//
// A started foreground service keeps the process in the perceptible set for
// exactly as long as the server runs. The service class, its dataSync type and
// the notification permission are already declared in the Android manifest;
// Wails v3 exposes StartForegroundService/StopForegroundService, and both are
// fire-and-forget JNI calls (the bridge no-ops when the Java side is not
// reachable), so they can never fail a start or a stop.

import "strconv"

// keepAwakeOnFn / keepAwakeOffFn are the seams the llama-server lifecycle
// calls (same style as pathsAndroidFilesDir): production binds the platform
// implementation, the desktop test binary records the calls. Both are no-ops
// off-Android (see foreground_other.go), so the call sites need no OS branch.
var (
	keepAwakeOnFn  = keepAwakeOn
	keepAwakeOffFn = keepAwakeOff
)

// servingNotification builds the ongoing-notification copy: the resident model
// names what is being served in direct mode (Android always has one), the
// plain statement covers every other start.
func servingNotification(resident string) (string, string) {
	title := tr("MyLlama 正在运行推理服务", "MyLlama is running an inference service")
	if resident == "" {
		return title, tr("llama-server 正在后台运行", "llama-server is running in the background")
	}
	return title, resident
}

// foregroundPayload renders the {"title","text"} object the Java bridge parses.
// strconv.Quote yields valid JSON string literals for these values (printable
// non-ASCII is carried through as UTF-8), so the builder cannot fail.
func foregroundPayload(title, text string) string {
	return `{"title":` + strconv.Quote(title) + `,"text":` + strconv.Quote(text) + `}`
}
