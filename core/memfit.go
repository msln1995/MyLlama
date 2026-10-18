package core

import "fmt"

// ─── Direct-mode memory pre-check ─────────────────────────────────
// Android has no swap and runs a low-memory killer: when a model's weights
// alone exceed the memory the kernel reports as available, loading it does not
// fail as an error — the process thrashes until the app (and its WebView
// renderer) is killed, which reads as an unexplained crash. Comparing them up
// front turns that silent kill into a message naming both numbers. Weights
// only: the KV cache and the compute buffers are left out of the comparison so
// a start that could have worked is never refused.

// probeFreeMemoryGB is the available-memory seam (getFreeMemoryGB in
// production), letting tests drive both the fit and the probe-failure path.
var probeFreeMemoryGB = getFreeMemoryGB

// availableMemoryBytes returns the memory a load may ask for, in bytes; 0 when
// the probe yields nothing.
func availableMemoryBytes() int64 {
	gb := probeFreeMemoryGB()
	if gb <= 0 {
		return 0
	}
	return int64(gb * float64(1<<30))
}

// memoryFitError returns nil when weightsBytes fits in availBytes and a
// bilingual error carrying both measured figures when it cannot. An unknown
// availBytes (failed probe) never blocks a start. Pure, so tests can walk the
// boundary without a device.
func memoryFitError(name string, weightsBytes, availBytes int64) error {
	if availBytes <= 0 || weightsBytes <= availBytes {
		return nil
	}
	return fmt.Errorf(tr("模型 %s 的权重就有 %s，本机当前只剩 %s 可用内存，放不下（加载时会被系统查杀、应用闪退）。请换更小的量化（如 Q4_K_M）或改用手机尺寸的模型。",
		"model %s weighs %s but only %s of memory is available right now, so it cannot fit (the load gets killed by the system and the app disappears). Pick a smaller quantization (e.g. Q4_K_M) or a phone-sized model."),
		name, formatBytes(weightsBytes), formatBytes(availBytes))
}

// modelLoadFitError checks one scanned model against the memory available
// right now, using the same weight figure the auto-tuner plans with
// (tuneWeightsBytes, which adds a same-directory mmproj projector).
func modelLoadFitError(m ModelInfo) error {
	return memoryFitError(m.Name, tuneWeightsBytes(m), availableMemoryBytes())
}
