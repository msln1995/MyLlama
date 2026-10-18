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
// files (tuneWeightsBytes adds the one mmproj projector cfg actually loads, the
// same file the preset passes to llama-server), the KV cache
// cfg.CtxSize needs, and the runtime overhead the auto-tuner plans with
// (batch compute buffers, graph scratch, sampler arenas) priced at cfg's own
// ubatch — the term is linear in ubatch, so a user who raises --ubatch-size for
// faster prefill is charged for the memory it actually costs.
//
// The KV figure reuses the same tuneModel geometry the auto-tuner sizes its
// context against and prices it with the cache type the config actually selects
// (kvCacheSizeFactor's measured table), so the guard refuses exactly the plans
// that budget cannot pay for. Pricing every selection but a pinned q8_0 pair at
// f16 was not the conservatism it looked like: it charged a q4_0 pair 3.6 times
// what llama.cpp allocates for it, which both refused loads that fit and made
// fittingCtxRung name a tier two rungs below the one the user could afford. Only
// the q8_0 pair was ever honored because it was the only selection with a
// measured footprint ratio - the table now has one for every type the settings
// page offers, on the same backend this guard runs on. Both serving modes pass
// --cache-type-k/v through, so the halved cache is real here. cfg.CtxSize <= 0
// charges no KV; the router-mode callers price the same models through
// routerModelPlan, which bounds an untuned context first, so nothing that
// reaches this function is actually unbounded.
func modelLoadBytes(m ModelInfo, cfg ModelConfig) int64 {
	mm, ok := readGGUFModelMetrics(m.Path)
	tm := buildTuneModel(mm, ok, tuneWeightsBytes(m, cfg))
	total := tm.WeightsBytes
	if cfg.CtxSize > 0 {
		kv := tm.KVBytesPerTokPerLayerF16 * float64(tm.Layers) * float64(cfg.CtxSize)
		total += int64(kv * kvCacheSizeFactor(cfg))
	}
	return total + runtimeOverheadBytes(cfg.UBatchSize)
}

// fittingCtxRung is the advice half of the memory check: the largest tier of the
// auto-tuner's own context ladder that this model, priced with cfg's other
// settings (KV cache types, ubatch), would fit in availBytes - and that is
// smaller than the context the user has configured now. 0 means nothing on the
// ladder helps: the weights and the runtime overhead alone already exceed the
// memory available, or the current context is at or below the ladder's floor.
//
// The refusal message carries this number because the two figures alone are not
// actionable on a phone: "needs 8.5 GB, 7.0 GB free" tells the user nothing they
// can type, while the tier the same pricing approves is. It never suggests
// raising the context - a rung above the configured one prices higher than the
// refused load (the price only grows with the context), so if one fitted, the
// refusal itself would be impossible.
//
// Each rung is priced by modelLoadBytes, which re-reads the GGUF header; this
// runs on a load that is about to be refused, not on a successful start.
func fittingCtxRung(m ModelInfo, cfg ModelConfig, availBytes int64) int {
	if availBytes <= 0 {
		return 0
	}
	mm, _ := readGGUFModelMetrics(m.Path)
	for _, ctx := range ctxLadderFor(mm.ContextLength, tuneCtxLadder) {
		if ctx >= cfg.CtxSize {
			continue
		}
		c := cfg
		c.CtxSize = ctx
		if modelLoadBytes(m, c) <= availBytes {
			return ctx
		}
	}
	return 0
}

// ─── Prompt-cache budget (both serving modes) ─────────────────────

// promptCacheShare is how much of the memory the planned load leaves over the
// prompt cache may retain. The rest stays with the OS: a machine that begins a
// session with zero slack has nothing left for the next turn's allocator, and on
// a phone the low-memory killer is the fallback.
const promptCacheShare = 0.5

// routerLibrary snapshots the two inputs every launch-time memory decision needs
// - the scanned model list and the stored per-model configs - so the prompt-cache
// bound and the over-commit warning price the same library.
func routerLibrary() ([]ModelInfo, map[string]ModelConfig) {
	models := scanModels()
	modelConfigsMu.Lock()
	cfgs := make(map[string]ModelConfig, len(cachedModelConfigs))
	for k, v := range cachedModelConfigs {
		cfgs[k] = v
	}
	modelConfigsMu.Unlock()
	return models, cfgs
}

// heaviestPlannedLoad is the largest load a router-mode server may bring up
// during its lifetime, together with the model it belongs to. Models load lazily
// on their first request, so the start-time prompt-cache budget cannot be priced
// against what is resident at launch (nothing) - it has to assume the heaviest
// entry in the library gets loaded at some point in the session.
//
// The whole load is charged against host RAM, KV cache included. That is not a
// conservative guess on the platforms this tuner plans for: on Apple unified
// memory the offloaded KV is host memory, measured as a flat 112.0 KiB per token
// (RSS 1,371 -> 4,060 MiB between ctx 8192 and ctx 32768 with -ngl 99 and the
// cache disabled, for a model whose f16 KV geometry is 112 KiB/token). Only a
// discrete-VRAM machine (CUDA) over-charges here, and over-charging is the safe
// direction for a bound that only ever shrinks a budget - it is not safe for a
// warning, which is why overcommitWarning asks first.
//
// (zero values, 0) for an empty or unreadable library. The model and its plan
// come back with the figure because the over-commit warning advises that one
// model, and advising it needs the plan the price was computed from.
func heaviestPlannedLoad(models []ModelInfo, cfgs map[string]ModelConfig) (ModelInfo, ModelConfig, int64) {
	var heavy ModelInfo
	var plan ModelConfig
	var heaviest int64
	for _, m := range models {
		c := routerModelPlan(cfgs, m)
		if n := modelLoadBytes(m, c); n > heaviest {
			heaviest, heavy, plan = n, m, c
		}
	}
	return heavy, plan, heaviest
}

// maxModelLoadBytes is heaviestPlannedLoad's byte figure alone.
func maxModelLoadBytes(models []ModelInfo, cfgs map[string]ModelConfig) int64 {
	_, _, n := heaviestPlannedLoad(models, cfgs)
	return n
}

// probeServingGPU is the serving-GPU seam for the router-mode over-commit
// warning (servingGPUPlan in production) so tests can drive the host-RAM gate
// without a GPU.
var probeServingGPU = servingGPUPlan

// servingGPUPlan resolves the GPU llama-server will serve on: the same plan
// target the auto-tuner sizes its contexts against, read from the cached probe
// snapshot (one detection chain per process, no extra nvidia-smi at start).
func servingGPUPlan() tuneTarget {
	serverConfigMu.Lock()
	deviceID := cachedServerConfig.DeviceID
	serverConfigMu.Unlock()
	return tunePlanTarget(gpuListSource(), systemInfo().CUDA.Available, deviceID)
}

// loadsStayInHostRAM reports whether modelLoadBytes' figure, which charges the
// whole load (weights and KV alike) against host RAM, describes what the machine
// really runs out of memory for. It does on Apple Silicon, where -ngl 99 puts
// the cache in main memory (measured: RSS grows by exactly the f16 KV geometry,
// 112.0 KiB per token), and on a CPU-only host. It does not on a machine with a
// discrete GPU big enough to offload into, where the same figure over-charges -
// acceptable for a bound that only ever shrinks a budget, not for a warning that
// would be wrong. A GPU under tuneGPUActiveMinVRAMMB counts as host-bound for the
// tuner's own reason: it cannot hold the load, so the load stays in RAM anyway.
func loadsStayInHostRAM(serving tuneTarget) bool {
	return serving.Vendor == vendorApple ||
		serving.Vendor == vendorNone ||
		serving.VRAMMB < tuneGPUActiveMinVRAMMB
}

// overcommitWarning is the router-mode counterpart of direct mode's
// modelLoadFitError, for the platform that cannot refuse: router mode is lazy
// and multi-model, so a library whose heaviest entry is unaffordable still has
// every other model to serve and the start must not fail.
//
// It is not a hypothetical. Measured on a 24 GiB M2 with one model pinned at an
// explicit ctx-size the machine cannot pay for: the load was accepted without an
// error, RSS reached 15.85 GiB, swap passed 7 GiB, and the request simply waited
// with no diagnostic anywhere - the whole machine ground while llama-server
// printed nothing. Saying which model and both figures is the only way that
// state is discoverable from the app.
//
// "" when the library is empty or unreadable, the probe failed (availBytes <= 0
// proves nothing), or the heaviest load fits. suggestCtx is fittingCtxRung's
// answer for that model (>0 = the tier to move to, 0 = no tier on the ladder
// helps), computed by the caller so this stays a pure formatter.
func overcommitWarning(name string, needBytes, availBytes int64, suggestCtx int) string {
	if name == "" || availBytes <= 0 || needBytes <= availBytes {
		return ""
	}
	advice := tr("请在模型设置里调小它的上下文大小，或用一键调优重新规划。",
		"Lower that model's context size in its settings, or run one-click tuning to re-plan it.")
	if suggestCtx > 0 {
		advice = fmt.Sprintf(tr("把它的上下文大小调到 %d 就放得下，或者用一键调优重新规划。",
			"Set its context size to %d and it fits, or run one-click tuning to re-plan it."), suggestCtx)
	}
	return fmt.Sprintf(tr("模型 %s 这次要占约 %s（权重 + 上下文缓存 + 运行开销），而本机当前只有 %s 空闲内存。服务仍会启动、其他模型照常可用，但真正加载它时内存放不下会开始交换：请求可能长时间没有响应、整台机器变卡。%s",
		"model %s needs about %s for this load (weights + context cache + runtime overhead) but only %s of memory is free right now. The service still starts and every other model stays usable, but loading it will overflow into swap: requests may hang for a long time and the whole machine slows down. %s"),
		name, formatBytes(needBytes), formatBytes(availBytes), advice)
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
// it out. Leaving it out does not save anything: the server sizes the cache
// itself, from the model's metadata and its own notion of what the machine may
// spend, and the app gets neither the number nor the bill. Measured with
// llama.cpp b10537 on a 24 GiB M2:
//
//	                             n_ctx_slot    RSS (router + worker)
//	Qwen3-0.6B Q4_K_M, no ctx-size        40,960       4.86 GiB
//	same model, ctx-size = 40960          40,960       4.84 GiB
//	same model, ctx-size = 32768          32,768       3.98 GiB
//	Qwen3-4B-Instruct-2507 Q4_K_M, no ctx-size 103,936  17.0 GiB (twice: 17,013 / 17,062 MiB)
//	same model, ctx-size = 16384          16,384        4.83 GiB (twice: 4,832 / 4,832 MiB)
//
// The 0.6B pair shows the omitted number costs what it states (its metadata
// context, 40,960 tokens, rung for rung: the 8,192 tokens between 32,768 and
// 40,960 are 0.88 GiB of RSS, the model's 112 KiB/token f16 geometry), and the
// 4B pair shows what an unbounded pick really means — the server took 103,936
// tokens, 17 GiB of RSS and swap 3.86 -> 6.30 GiB, for a model the app had
// already priced past the machine.
//
// That over-allocation is the smaller half of the harm. The library's
// prompt-cache budget is sized against the heaviest planned load
// (heaviestPlannedLoad), and until this bound existed that load was priced at the
// model's trained context, so one unconfigured model was by itself enough to push
// the headroom negative and turn prompt caching off for every model in the app,
// in every session: on this M2 (7,925 MiB available at the time of the runs
// above) the 4B at its metadata 262,144 tokens prices at 40,525 MiB and bounds
// --cache-ram to 0, while the bounded 16,384 entry prices at 5,965 MiB and keeps
// 980 MiB.
//
// The rule keeps the model's own context whenever the app's load price fits the
// memory available now - never above tuneCtxMax, the ceiling every tuned plan is
// sized against, so an untuned 256k model cannot end up holding a bigger window
// than the tuned one would - and otherwise takes the largest rung of the tuner's
// ladder that fits, down to the ladder's floor. Unreadable metadata or a failed
// memory probe land on the settings page default, the same fallback the rest of
// this file uses.
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
	// The tuner's ceiling applies to untuned models too: a model trained at
	// 262,144 must not get a bigger window than the same model's tuned plan
	// would, on any machine.
	keep := mm.ContextLength
	if keep > tuneCtxMax {
		keep = tuneCtxMax
	}
	if modelLoadBytes(m, withCtx(keep)) <= avail {
		return keep
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
// without a device; suggestCtx (fittingCtxRung's answer, 0 = nothing on the
// ladder helps) is passed in for the same reason.
func memoryFitError(name string, needBytes, availBytes int64, suggestCtx int) error {
	if availBytes <= 0 || needBytes <= availBytes {
		return nil
	}
	advice := tr("最小的上下文档也放不下：这台机器需要一个更小的量化或更小的模型（或者把上下文调到 2048 以下）。",
		"even the smallest context tier does not fit: this device needs a smaller quantization or a smaller model (or a context below 2048 tokens).")
	if suggestCtx > 0 {
		advice = fmt.Sprintf(tr("把它的上下文大小调到 %d 就放得下，或者用一键调优重新规划。",
			"Set its context size to %d and it fits, or run one-click tuning to re-plan it."), suggestCtx)
	}
	return fmt.Errorf(tr("模型 %s 这次要占约 %s（权重 + 上下文缓存 + 运行开销），而本机当前只剩 %s 可用内存，放不下（加载时会被系统查杀、应用闪退）。%s",
		"model %s needs about %s for this load (weights + context cache + runtime overhead) but only %s of memory is available right now, so it cannot fit (the load gets killed by the system and the app disappears). %s"),
		name, formatBytes(needBytes), formatBytes(availBytes), advice)
}

// modelLoadFitError checks one scanned model against the memory available
// right now, with the same budget the auto-tuner sizes its context against, and
// names the context tier that would fit when one does.
func modelLoadFitError(m ModelInfo, cfg ModelConfig) error {
	avail := availableMemoryBytes()
	return memoryFitError(m.Name, modelLoadBytes(m, cfg), avail, fittingCtxRung(m, cfg, avail))
}
