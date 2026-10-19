package core

import (
	"fmt"
	"sort"
	"strconv"
)

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

// ctxCheckpointCap is the llama-server context-checkpoint budget every plan
// that does not pin them off runs with, in place of upstream's default of 32.
//
// Context checkpoints are a host-RAM retention the guard used to miss entirely.
// A model whose memory cannot be pruned partially — sliding-window models
// (n_swa > 0 without --swa-full) and the hybrid/recurrent ones llama.cpp
// classifies as FULL/RS sequence-removal — copies that state into a checkpoint
// while processing a prompt, up to the budget, and the server restores one
// automatically when a request shares a long prefix with the cached prompt but
// diverges before its end (an edited or regenerated turn). Plain-attention
// models roll back in place and checkpoint nothing: measured against upstream
// b11720, Qwen3-4B Q4_K_M given a single 22,321-token prompt logged
// "context checkpoints enabled, max = 32" and created zero of them.
//
// The price of one checkpoint is the sliding-window slice of the KV cache.
// Measured with gemma-3-1b-it Q4_K_M (26 layers, 22 of them sliding-window at
// window 512, 1 KV head of 256 f16 = 1024 B per layer per token): every logged
// checkpoint was 11.006 MiB, and 22 x 1024 x 512 = 11.0 MiB — the guard's
// all-layer figure (26 x 1024 x 512 = 13.0 MiB) over-charges by 18%, which is
// the direction a start-time budget wants its error in. With the q8_0 K/V pair
// the same checkpoints came out at 5.85 MiB, i.e. kvCacheSizeFactor's measured
// 0.531 again, so the cache type is charged with the same table the KV term
// uses.
//
// 8 rather than 32 because the retention is what scales and the benefit
// saturates at once. Same fixture, ctx 16384, four requests walking a shared
// 13,553-token prefix and then revisiting a branched one:
//
//	checkpoints   peak RSS   revisit prefill   forced full re-processed
//	32 (default)  1.21 GiB   2,724 tok / 2.24 s       0
//	8             1.09 GiB   2,724 tok / 2.26 s       0
//	0             1.03 GiB  10,861 tok / 8.39 s       2
//
// The cap keeps the whole saving (the list is oldest-first, and the search reads
// it newest-first, so a short list still holds the state a chat revisits), and
// append-only conversation — what the app's own chat page sends — never uses a
// checkpoint at all (six appended turns ran identically with them off).
const ctxCheckpointCap = 8

// ctxCheckpointValue is the number emitted for cfg's plan: 0 on Android (where
// every plan pins them off and the load is memory-existential) and for any plan
// the tuner pinned off, else the cap. modelPresetKV writes exactly this value and
// ctxCheckpointBytes charges exactly the memory it buys, so the preset and the
// guard cannot drift apart.
func ctxCheckpointValue(cfg ModelConfig) int {
	if cfg.CtxCheckpointsOff || platformGOOS == "android" {
		return 0
	}
	return ctxCheckpointCap
}

// ctxCheckpointBytes prices what the checkpoint budget can hold for one load:
// cap x the model's per-token KV total x the window it copies x the configured
// cache type, clamped by the context (a checkpoint cannot span more than the
// window the load runs with). 0 when the plan runs no checkpoints, when the
// model has no sliding window (nothing is checkpointed), or when no context is
// configured yet.
//
// Recurrent-state models are the known under-charge: their checkpoint holds the
// recurrent state rather than a KV slice, and no KV-times-window formula
// describes it. They are charged 0 here, which is what they were charged before
// this term existed, but they now run at the cap instead of 32, so the same
// change that adds the price also shrinks that unpriced term fourfold.
func ctxCheckpointBytes(tm tuneModel, cfg ModelConfig) int64 {
	n := ctxCheckpointValue(cfg)
	if n == 0 || tm.SwaWindow <= 0 || cfg.CtxSize <= 0 {
		return 0
	}
	window := tm.SwaWindow
	if window > cfg.CtxSize {
		window = cfg.CtxSize
	}
	perTok := tm.KVBytesPerTokPerLayerF16 * float64(tm.Layers) * kvCacheSizeFactor(cfg)
	return int64(float64(n) * perTok * float64(window))
}

// modelLoadBytes estimates what loading m with cfg keeps resident: the weight
// files (tuneWeightsBytes adds the one mmproj projector cfg actually loads, the
// same file the preset passes to llama-server), the KV cache
// cfg.CtxSize needs, the context-checkpoint budget the plan runs with
// (ctxCheckpointBytes — a copy of the sliding-window slice of that same cache,
// which is why the user-facing messages count it under 上下文缓存 / "context
// cache" rather than as a fourth category), and the runtime overhead the
// auto-tuner plans with (batch compute buffers, graph scratch, sampler arenas)
// priced at cfg's own ubatch in the shape the load runs (loadRuntimeOverheadBytes
// — the term is linear in ubatch, so a user who raises --ubatch-size for faster
// prefill is charged for the memory it actually costs).
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
	total += ctxCheckpointBytes(tm, cfg)
	return total + loadRuntimeOverheadBytes(cfg, probeServingGPU(), tm.Layers)
}

// loadRuntimeOverheadBytes is the non-weight, non-KV reserve term of
// modelLoadBytes, taken in the shape the load actually runs: the Metal
// measurement when llama-server serves this on Apple silicon's unified GPU with
// every layer offloaded, the CPU-only (phone-measured) figure otherwise.
//
// The serving GPU is the same cached probe the over-commit warning next to it
// reads (probeServingGPU, one detection chain per process), so the two memory
// decisions on this path cannot disagree about which machine they are pricing,
// and the platform has to say so too: the GPU list is arch- and OS-gated, so
// darwin + apple is the only way this shape exists, and a snapshot left over from
// another platform can never reprice a phone's load.
//
// An unpinned config is the Metal shape too, not a guess at it: llama-server's own
// default offloads every layer on Apple silicon, measured as the same residual
// this term prices (+141 MiB beyond weights + f16 KV for a 4B Q4_K_M at ctx 4096,
// against +2585 MiB of RSS with -ngl 0, most of which is the model's own file
// pages - and those the available-memory probe this term is compared against
// counts as reclaimable). What that residual tracks is the layers the host still
// computes, so the shape follows offloadsEveryLayer rather than the mere absence
// of a -ngl 0.
func loadRuntimeOverheadBytes(cfg ModelConfig, serving tuneTarget, layers int) int64 {
	if platformGOOS == "darwin" && serving.Vendor == vendorApple && offloadsEveryLayer(cfg.GPULayers, layers) {
		return metalRuntimeOverheadBytes(cfg.UBatchSize)
	}
	return runtimeOverheadBytes(cfg.UBatchSize)
}

// offloadsEveryLayer reports whether a gpu-layers setting leaves no layer on the
// CPU, which is what decides the reserve shape on Apple silicon. The term is not a
// property of the chip - it tracks the layers the host still computes, measured on
// the M2 with a 36-layer 4B Q4_K_M at ctx 8192 and ubatch 512 (paired runs):
// -ngl 99 +141/+142, -ngl 27 (9 layers on the CPU) +727/+728, -ngl 18 +1232/+1234,
// -ngl 0 +2504. Roughly 63 MiB per CPU-resident layer on top of the ~140 the GPU
// run keeps, so anything short of full offload pays the CPU figure; that
// over-charges a 30-of-36 layer split by a few hundred MiB, which is the safe
// direction for a reserve.
//
// "" / "auto" / "all" are the full-offload spellings (the first two emit no -ngl
// flag, whose server-side default is full offload here). A number counts only when
// the model's layer count is known and it reaches it - so an unreadable header's
// fallback count (32) decides, never a silent "unknown means Metal".
func offloadsEveryLayer(gpuLayers string, layers int) bool {
	switch gpuLayers {
	case "", "auto", "all":
		return true
	}
	n, err := strconv.Atoi(gpuLayers)
	return err == nil && layers > 0 && n >= layers
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

// plannedLoad is one library entry with the plan a router-mode load runs with
// and the memory that load prices (modelLoadBytes).
type plannedLoad struct {
	model ModelInfo
	cfg   ModelConfig
	bytes int64
}

// residentSet is what router mode may hold in memory at the same time.
type residentSet struct {
	// loads are the entries that can be resident together, heaviest first,
	// capped at the server's max-loaded-models count.
	loads []plannedLoad
	// total is the sum of loads: the figure a session can drive the machine to.
	total int64
	// heaviest is the single most expensive load, the entry an over-commit
	// warning names as the one to shrink.
	heaviest plannedLoad
}

// planResidentSet prices the loads a router-mode server may keep resident at
// once: the modelsMax heaviest entries in the library.
//
// Models load lazily on their first request, so a start-time decision cannot
// price against what is resident at launch (nothing) - it has to assume the
// expensive entries get loaded at some point in the session. The cap is a count
// rather than a memory bound, and the loads really do stack. Measured with
// llama.cpp b11720 on a 24 GiB M2 (a four-id --models-preset, --cache-ram 0,
// one worker process per loaded model): with --models-max 2 and the requests
// addressed big -> small-a -> small-b -> small-c, RSS across all llama-server
// processes walked
//
//	109 MiB (nothing loaded) -> 3,211 (big) -> 3,663 (big + small-a)
//	-> 1,013 (small-a + small-b, big evicted) -> 1,010 (small-b + small-c)
//
// — two loads resident at once, the next request evicting the least recently
// used one, so a session's peak is the sum of the modelsMax most expensive
// loads, never the heaviest of them alone. Pricing only the heaviest is what
// this file used to do, which understated the bill by one whole load per extra
// slot whenever the user raised the app's "max loaded models" setting above its
// default of 1 (the app's own docs describe that setting as the number of
// models resident in memory at the same time).
//
// Each load is charged against host RAM in full, KV cache included. That is not
// a conservative guess on the platforms these decisions serve: on Apple unified
// memory the offloaded KV is host memory, measured as a flat 112.0 KiB per token
// (RSS 1,371 -> 4,060 MiB between ctx 8192 and ctx 32768 with -ngl 99 and the
// cache disabled, for a model whose f16 KV geometry is 112 KiB/token). Only a
// discrete-VRAM machine (CUDA) over-charges here, and over-charging is the safe
// direction for a bound that only ever shrinks a budget - it is not safe for a
// warning, which is why overcommitWarning asks first.
//
// modelsMax <= 1 (the app's default) yields the single heaviest entry, exactly
// the figure these decisions used. An empty or unreadable library yields a
// zero set.
func planResidentSet(models []ModelInfo, cfgs map[string]ModelConfig, modelsMax int) residentSet {
	if modelsMax < 1 {
		modelsMax = 1
	}
	loads := make([]plannedLoad, 0, len(models))
	for _, m := range models {
		c := routerModelPlan(cfgs, m)
		loads = append(loads, plannedLoad{model: m, cfg: c, bytes: modelLoadBytes(m, c)})
	}
	sort.SliceStable(loads, func(i, j int) bool { return loads[i].bytes > loads[j].bytes })
	set := residentSet{}
	if len(loads) == 0 {
		return set
	}
	set.heaviest = loads[0]
	if len(loads) > modelsMax {
		loads = loads[:modelsMax]
	}
	set.loads = loads
	for _, l := range loads {
		set.total += l.bytes
	}
	return set
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
// proves nothing), or the priced loads fit. suggestCtx is fittingCtxRung's
// answer for the heaviest model (>0 = the tier to move to, 0 = no tier on the
// ladder helps), computed by the caller so this stays a pure formatter.
//
// resident is how many loads totalBytes prices (planResidentSet's set size) and
// heaviestBytes is that one model's own share of them. At resident == 1 the two
// are the same load and the sentence stays the single-load one; above 1 it says
// how many can be resident and names the cap as the other lever, because the
// number the user can actually type is what makes the line actionable.
func overcommitWarning(name string, resident int, totalBytes, heaviestBytes, availBytes int64, suggestCtx int) string {
	if name == "" || availBytes <= 0 || totalBytes <= availBytes {
		return ""
	}
	advice := tr("请在模型设置里调小它的上下文大小，或用一键调优重新规划。",
		"Lower that model's context size in its settings, or run one-click tuning to re-plan it.")
	if suggestCtx > 0 {
		advice = fmt.Sprintf(tr("把它的上下文大小调到 %d 就放得下，或者用一键调优重新规划。",
			"Set its context size to %d and it fits, or run one-click tuning to re-plan it."), suggestCtx)
	}
	// The tier above is priced with the other resident loads still in memory
	// (the caller hands fittingCtxRung the headroom they leave over), so when a
	// tier exists it really is the number that fixes this. When none does, the
	// cap is the lever left to pull.
	if resident > 1 && suggestCtx <= 0 {
		advice = tr("把「最大加载模型数」调小，或在模型设置里调小它的上下文大小、用一键调优重新规划。",
			"Lower the max-loaded-models count, or shrink this model's context in its settings and re-plan it with one-click tuning.")
	}
	if resident > 1 {
		return fmt.Sprintf(tr("最多 %d 个模型会同时驻留，合计要占约 %s（每个模型各自的权重 + 上下文缓存 + 运行开销）；其中 %s 最重，它一项就要占约 %s。而本机当前只有 %s 空闲内存。服务仍会启动、其他模型照常可用，但真加载到这么多时内存放不下会开始交换：请求可能长时间没有响应、整台机器变卡。%s",
			"up to %d models can stay resident at once, which costs about %s in total (each model's own weights + context cache + runtime overhead); %s is the heaviest and takes about %s by itself. Only %s of memory is free right now. The service still starts and every other model stays usable, but loading that many will overflow into swap: requests may hang for a long time and the whole machine slows down. %s"),
			resident, formatBytes(totalBytes), name, formatBytes(heaviestBytes), formatBytes(availBytes), advice)
	}
	return fmt.Sprintf(tr("模型 %s 这次要占约 %s（权重 + 上下文缓存 + 运行开销），而本机当前只有 %s 空闲内存。服务仍会启动、其他模型照常可用，但真正加载它时内存放不下会开始交换：请求可能长时间没有响应、整台机器变卡。%s",
		"model %s needs about %s for this load (weights + context cache + runtime overhead) but only %s of memory is free right now. The service still starts and every other model stays usable, but loading it will overflow into swap: requests may hang for a long time and the whole machine slows down. %s"),
		name, formatBytes(totalBytes), formatBytes(availBytes), advice)
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
// prompt-cache budget is sized against the loads that can be resident at once
// (planResidentSet, at the app's default cap of one model the heaviest planned
// load), and until this bound existed that load was priced at the
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
