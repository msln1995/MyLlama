package core

import "fmt"

// ─── Memory fit pre-check (direct mode) ───────────────────────────
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
// files (tuneWeightsBytes adds a same-directory mmproj projector), the KV cache
// cfg.CtxSize needs, and the runtime overhead the auto-tuner plans with
// (batch compute buffers, graph scratch, sampler arenas) priced at cfg's own
// ubatch — the term is linear in ubatch, so a user who raises --ubatch-size for
// faster prefill is charged for the memory it actually costs.
//
// The KV figure reuses the same tuneModel geometry the auto-tuner sizes its
// context against, so the guard refuses exactly the plans that budget cannot
// pay for. The q8_0 pair is priced at tuneQ8KVSizeRatio because that is the one
// cache selection the RAM-bounded plans keep and price the same way (see
// tuneModel.PinnedQ8KV): pricing it at f16 instead would have this guard refuse
// the very context the tuner just selected, and direct mode does pass
// --cache-type-k/v through, so the halved cache is real here. Every other cache
// type stays charged at f16, the upper bound, since a quantized cache only makes
// the load smaller and only q8_0 has a measured footprint ratio on this tuner's
// hardware. cfg.CtxSize <= 0 charges no KV, matching the preset writer, which
// omits --ctx-size there.
func modelLoadBytes(m ModelInfo, cfg ModelConfig) int64 {
	mm, ok := readGGUFModelMetrics(m.Path)
	tm := buildTuneModel(mm, ok, tuneWeightsBytes(m))
	total := tm.WeightsBytes
	if cfg.CtxSize > 0 {
		kv := tm.KVBytesPerTokPerLayerF16 * float64(tm.Layers) * float64(cfg.CtxSize)
		if pinnedQ8KV(cfg) {
			kv *= tuneQ8KVSizeRatio
		}
		total += int64(kv)
	}
	return total + runtimeOverheadBytes(cfg.UBatchSize)
}

// ─── Prompt-cache budget (both serving modes) ─────────────────────

// promptCacheShare is how much of the memory the planned load leaves over the
// prompt cache may retain. The rest stays with the OS: a machine that begins a
// session with zero slack has nothing left for the next turn's allocator, and on
// a phone the low-memory killer is the fallback.
const promptCacheShare = 0.5

// heaviestRouterLoad estimates the largest load a router-mode server may bring
// up during its lifetime. Models load lazily on their first request, so the
// start-time prompt-cache budget cannot be priced against what is resident at
// launch (nothing) — it has to assume the heaviest entry in the library gets
// loaded at some point in the session. 0 for an empty or unreadable library.
//
// The whole load is charged against host RAM, KV cache included. That is not a
// conservative guess on the platforms this tuner plans for: on Apple unified
// memory the offloaded KV is host memory, measured as a flat 112.0 KiB per token
// (RSS 1,371 -> 4,060 MiB between ctx 8192 and ctx 32768 with -ngl 99 and the
// cache disabled, for a model whose f16 KV geometry is 112 KiB/token). Only a
// discrete-VRAM machine (CUDA) over-charges here, and over-charging is the safe
// direction because the bound only ever shrinks a budget.
func heaviestRouterLoad() int64 {
	models := scanModels()
	modelConfigsMu.Lock()
	cfgs := make(map[string]ModelConfig, len(cachedModelConfigs))
	for k, v := range cachedModelConfigs {
		cfgs[k] = v
	}
	modelConfigsMu.Unlock()
	return maxModelLoadBytes(models, cfgs)
}

// maxModelLoadBytes is heaviestRouterLoad's pure core: the largest modelLoadBytes
// over the given models and per-model configs. Pure so tests drive the
// library-shape cases (which config wins, the untuned fallback, an empty list)
// without a device or a model directory.
func maxModelLoadBytes(models []ModelInfo, cfgs map[string]ModelConfig) int64 {
	var heaviest int64
	for _, m := range models {
		if n := modelLoadBytes(m, routerModelPlan(cfgs, m)); n > heaviest {
			heaviest = n
		}
	}
	return heaviest
}

// routerModelPlan resolves the config a router-mode load runs m with: the stored
// per-model config when there is one, otherwise the untuned entry the preset
// writer emits (see routerUntunedCtx). A model whose stored config carries no
// context is treated as untuned here too, so the price and the preset always
// describe the same load — the one number the library's prompt-cache budget is
// sized against.
func routerModelPlan(cfgs map[string]ModelConfig, m ModelInfo) ModelConfig {
	c, ok := cfgs[m.Name]
	if !ok {
		c = defaultModelConfig()
	}
	if !ok || c.CtxSize <= 0 {
		c.CtxSize = routerUntunedCtx(m)
	}
	return c
}

// routerUntunedCtx picks the context an unconfigured model gets in router mode,
// and the preset writer now writes that number into the entry instead of leaving
// it out. Omitting it is not a saving: llama-server's default is "0 = the context
// the model was trained with" and it really does build that cache. Measured on a
// 24 GiB M2 with Qwen3-0.6B Q4_K_M (378 MiB of weights, 112 KiB of f16 KV per
// token), an entry with no ctx-size came up as "n_ctx_slot = 40960" — the context
// its GGUF metadata records — with a 4.86 GiB worker RSS, indistinguishable from
// the same model whose entry states ctx-size = 40960 (4.84 GiB), while the
// ctx-size = 32768 entry cost 3.98 GiB. The 0.88 GiB between those two runs is
// the 8,192 tokens of context they differ by, at the f16 geometry, so the
// allocation tracks the trained context rung for rung.
//
// The unbounded allocation is the smaller half of the harm. The library's
// prompt-cache budget is sized against the heaviest planned load
// (maxModelLoadBytes), which already priced an unconfigured model at its trained
// context — so one model whose context exceeds the machine is by itself enough to
// push the headroom negative and turn prompt caching off for every model in the
// app, in every session. On an 8,218 MiB host an entry priced at the 262,144
// tokens a Qwen3-4B-Instruct-2507 records in its own metadata costs 30,330 MiB
// and bounds --cache-ram to 0; the same machine keeps 1,021 MiB once the entry
// states a context it can pay for.
//
// The rule keeps the model's own context whenever the app's load price fits the
// memory available now (so an untuned model on a roomy machine behaves as it does
// today, just stated explicitly and priced at what the server will build), and
// otherwise takes the largest rung of the tuner's ladder that fits, down to the
// ladder's floor. Unreadable metadata or a failed memory probe land on the
// settings page default, the same fallback the rest of this file uses.
func routerUntunedCtx(m ModelInfo) int {
	fallback := defaultModelConfig()
	mm, ok := readGGUFModelMetrics(m.Path)
	if !ok || mm.ContextLength <= 0 {
		return fallback.CtxSize
	}
	avail := availableMemoryBytes()
	if avail <= 0 {
		return fallback.CtxSize
	}
	withCtx := func(ctx int) ModelConfig {
		c := fallback
		c.CtxSize = ctx
		return c
	}
	if modelLoadBytes(m, withCtx(mm.ContextLength)) <= avail {
		return mm.ContextLength
	}
	for _, ctx := range ctxLadderFor(mm.ContextLength, tuneCtxLadder) {
		if modelLoadBytes(m, withCtx(ctx)) <= avail {
			return ctx
		}
	}
	return tuneCtxMin
}

// promptCacheRAMMiB bounds llama-server's prompt-cache budget (MiB) for one
// planned load. The cache is not a ceiling on something already allocated: every
// completed completion task copies the slot state into host RAM and keeps it
// until the budget forces an LRU eviction, so the process grows after the start
// guard has finished checking. Measured on an M2 with a 378 MiB Qwen3-0.6B
// Q4_K_M (112 KiB of f16 KV per token) serving ten distinct ~1,035-token
// requests: the server reported "saving prompt ... total state size = 113-114
// MiB" per request and RSS rose from 1,644 MiB to 2,796 MiB, tracking cached
// tokens x the KV geometry; the same requests with --cache-ram 0 left RSS flat
// (+41 MiB over 10,561 tokens), and --cache-ram 256 grew to a peak and fell back
// as it evicted. So the budget is a real cap and the retention is the load's
// only unbounded term.
//
// Re-measured at the desktop default of 8192 MiB with the same model serving six
// distinct ~6,890-token requests: +756 MiB of RSS per request and no plateau
// before the probe aborted at 5 GiB total — eleven requests of that size reach
// the full 8 GiB. That is why both serving modes bound the budget and not only
// the phone: a plan that already spends the RAM budget plus an 8 GiB cache is
// over-committed by design, and on a desktop it shows up as swapping mid-session
// instead of a killed process.
//
// A bound of 0 means "this load has no room left to retain anything", so callers
// must pass it through as --cache-ram 0 (disable): omitting the flag instead
// hands the budget to llama-server's own default, which is 8192 MiB — the exact
// opposite of what was measured.
//
// requestedMiB <= 0 (the cache is off) and an unknown availBytes (failed probe)
// both pass through untouched: this function only ever shrinks a budget the
// device demonstrably cannot carry. Pure, so tests drive the boundary without a
// device.
func promptCacheRAMMiB(requestedMiB int, plannedNeedBytes, availBytes int64) int {
	if requestedMiB <= 0 || availBytes <= 0 {
		return requestedMiB
	}
	headroom := availBytes - plannedNeedBytes
	if headroom <= 0 {
		return 0
	}
	bound := int64(float64(headroom) * promptCacheShare / float64(1<<20))
	if bound >= int64(requestedMiB) {
		return requestedMiB
	}
	return int(bound)
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
