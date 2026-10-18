package core

import "fmt"

// ─── Direct-mode memory pre-check ─────────────────────────────────
// Android has no swap and runs a low-memory killer: when a load asks for more
// memory than the kernel reports as available, it does not fail as an error —
// the process thrashes until the app (and its WebView renderer) is killed,
// which reads as an unexplained crash. Comparing the planned footprint against
// the memory available right now turns that silent kill into a message naming
// both numbers.

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

// modelLoadBytes estimates what loading m with cfg keeps resident: the weight
// files (tuneWeightsBytes adds a same-directory mmproj projector), the f16 KV
// cache cfg.CtxSize needs, and the runtime overhead the auto-tuner plans with
// (batch compute buffers, graph scratch, sampler arenas) priced at cfg's own
// ubatch — the term is linear in ubatch, so a user who raises --ubatch-size for
// faster prefill is charged for the memory it actually costs.
//
// The KV figure reuses the same tuneModel geometry the auto-tuner sizes its
// context against, so the guard refuses exactly the plans that budget cannot
// pay for. f16 is the upper bound: a quantized cache type only makes the load
// smaller. cfg.CtxSize <= 0 charges no KV, matching the preset writer, which
// omits --ctx-size there.
func modelLoadBytes(m ModelInfo, cfg ModelConfig) int64 {
	mm, ok := readGGUFModelMetrics(m.Path)
	tm := buildTuneModel(mm, ok, tuneWeightsBytes(m))
	total := tm.WeightsBytes
	if cfg.CtxSize > 0 {
		total += int64(tm.KVBytesPerTokPerLayerF16 * float64(tm.Layers) * float64(cfg.CtxSize))
	}
	return total + runtimeOverheadBytes(cfg.UBatchSize)
}

// memoryFitError returns nil when needBytes fits in availBytes and a bilingual
// error carrying both measured figures when it cannot. An unknown availBytes
// (failed probe) never blocks a start. Pure, so tests can walk the boundary
// without a device.
func memoryFitError(name string, needBytes, availBytes int64) error {
	if availBytes <= 0 || needBytes <= availBytes {
		return nil
	}
	return fmt.Errorf(tr("模型 %s 这次要占约 %s（权重 + 上下文缓存 + 运行开销），而本机当前只剩 %s 可用内存，放不下（加载时会被系统查杀、应用闪退）。请调小上下文大小，或换更小的量化（如 Q4_K_M）。",
		"model %s needs about %s for this load (weights + context cache + runtime overhead) but only %s of memory is available right now, so it cannot fit (the load gets killed by the system and the app disappears). Lower the context size, or pick a smaller quantization (e.g. Q4_K_M)."),
		name, formatBytes(needBytes), formatBytes(availBytes))
}

// modelLoadFitError checks one scanned model against the memory available
// right now, with the same budget the auto-tuner sizes its context against.
func modelLoadFitError(m ModelInfo, cfg ModelConfig) error {
	return memoryFitError(m.Name, modelLoadBytes(m, cfg), availableMemoryBytes())
}
