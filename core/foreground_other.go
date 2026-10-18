//go:build !android

package core

// Off-Android no-ops: a foreground service is an Android-only concept, and
// desktop processes are not frozen or cpuset-clipped when backgrounded. The
// llama-server lifecycle still calls through the keepAwake seams so the
// desktop test binary can assert the wiring.

func keepAwakeOn(title, text string) {}

func keepAwakeOff() {}
