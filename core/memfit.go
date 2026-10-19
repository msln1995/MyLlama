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
	total, _ := modelLoadPricing(m, cfg)
	return total
}

// modelLoadPricing prices a load the way modelLoadBytes does and also reports the
// share of that bill a Metal working set has to pin.
//
// The two figures differ only on a Mac, and only by hostWeightBytes - the weight
// copy a partial (or no) offload keeps on the host *in addition* to the copy the
// weight term already covers. Everything else in the bill is what the offloaded
// path allocates: the weights llama.cpp hands the device, the cache (measured
// host-resident RSS growing by exactly the f16 KV geometry at -ngl 99, i.e. the
// cache is the device's), the checkpoints and the Metal floor. The split is what
// lets the launch check name the right ceiling: a `-ngl 0` library priced past
// the pool touches no Metal at all, and a working-set warning about it would be
// wrong twice over.
//
// The weight term itself is not split by the offload line, so a *partial* offload
// is still billed to the pool in full: the figure over-prices the working set by
// the blocks the CPU runs instead of under-pricing it, which is the direction a
// ceiling that only ever shrinks a context or a cache has to take.
func modelLoadPricing(m ModelInfo, cfg ModelConfig) (total, pinned int64) {
	mm, ok := readGGUFModelMetrics(m.Path)
	tm := buildTuneModel(mm, ok, tuneWeightsBytes(m, cfg))
	total = tm.WeightsBytes
	if cfg.CtxSize > 0 {
		kv := tm.KVBytesPerTokPerLayerF16 * float64(tm.Layers) * float64(cfg.CtxSize)
		total += int64(kv * kvCacheSizeFactor(cfg))
	}
	total += ctxCheckpointBytes(tm, cfg)
	serving := probeServingGPU()
	total += loadRuntimeOverheadBytes(cfg, serving, tm)
	pinned = total
	if platformGOOS == "darwin" && serving.Vendor == vendorApple {
		if cpuResidentLayers(cfg.GPULayers, tm.Layers) >= tm.Layers {
			// Nothing was handed to the device: every layer, the cache and the
			// compute buffers stay on the CPU, so this load cannot be over pool.
			pinned = 0
		} else {
			pinned -= hostWeightBytes(cfg, tm)
		}
	}
	return total, pinned
}

// loadRuntimeOverheadBytes is the non-weight, non-KV reserve term of
// modelLoadBytes, taken in the shape the load actually runs: the CPU-only
// (phone-measured) figure everywhere, the Apple silicon one on a Mac serving from
// Metal.
//
// The serving GPU is the same cached probe the over-commit warning next to it
// reads (probeServingGPU, one detection chain per process), so the two memory
// decisions on this path cannot disagree about which machine they are pricing,
// and the platform has to say so too: the GPU list is arch- and OS-gated, so
// darwin + apple is the only way this shape exists, and a snapshot left over from
// another platform can never reprice a phone's load.
//
// On a Mac the term is the measured Metal floor plus the weights the host holds
// in addition to the copy the weight term already covers - see
// metalRuntimeOverheadBytes for the floor, hostWeightBytes for the second copy
// and cpuResidentLayers for the layer split. An unpinned config is the floor
// alone, not a guess at it: llama-server's own default offloads every layer on
// Apple silicon (measured: no -ngl flag at all leaves +141 MiB beyond weights +
// f16 KV, against +2504 MiB of RSS with -ngl 0).
func loadRuntimeOverheadBytes(cfg ModelConfig, serving tuneTarget, tm tuneModel) int64 {
	if platformGOOS != "darwin" || serving.Vendor != vendorApple {
		return runtimeOverheadBytes(cfg.UBatchSize)
	}
	return metalRuntimeOverheadBytes(cfg.UBatchSize) + hostWeightBytes(cfg, tm)
}

// hostWeightBytes prices the weights a Mac load keeps on the host *on top of* the
// weight term, which already covers one copy of the file. Two settings put bytes
// there, and each buys a second copy of a different part of the model:
//
//   - a gpu-layers number leaves the leading blocks to the CPU (it offloads the
//     last N), so the host holds those blocks' weights again;
//   - --cpu-moe / --n-cpu-moe moves the expert pool onto the CPU even for the
//     blocks the GPU runs, so the host holds those experts as well.
//
// Both were measured as phys_footprint (the dirty, non-reclaimable half) on the
// M2. The dense half is cpuResidentLayers' numbers. The expert half: OLMoE-1B-7B
// Q4_0 (16 blocks, 3746 MiB of weights of which 3456 MiB are experts) at ctx 4096
// and the default ubatch went from 575 MB of dirty memory with -ngl 99 to 4035 MB
// with -ngl 99 --cpu-moe - paired runs, the second copy of the reading inside
// 2 MB - a step of 3460 MiB against the 3456 MiB expert pool. Without the term
// this load was charged the 320 MiB floor alone, i.e. the guard under-priced it
// by 3.4 GB, exactly the error class that let a 15.85 GiB RSS thrash through as
// "fits".
//
// The two halves never overlap, and which blocks overlap is measured, not
// assumed: -ngl N offloads the *last* N blocks, so the host keeps the first
// L-N and the dense term already prices those; --n-cpu-moe K pins the *first* K
// blocks' experts, so the only bytes that move are the pinned blocks past the
// host cut line - max(0, K - host blocks). Measured on the same fixture at ctx
// 4096: --n-cpu-moe 4 with -ngl 12 read 1704 MB against 1706 MB without the flag,
// and --n-cpu-moe 8 and 12 with -ngl 4 both read 3505/3504 against 3504, i.e. no
// step at all while their pinned span sat inside the host's first twelve blocks;
// --n-cpu-moe 12 with -ngl 8 (host = first eight) added 650 MB, four blocks of
// experts, not the eight the reverse reading would give. Summed, the two terms
// are exactly the physical picture - all dense bytes for the host blocks plus the
// whole expert pool when the span covers every block - which is why a partial
// offload plus --cpu-moe does not pay for the same bytes twice (the naive "share
// of the file + share of the experts" would over-charge the 8-of-16 split on this
// fixture by 1.7 GiB).
func hostWeightBytes(cfg ModelConfig, tm tuneModel) int64 {
	layers := tm.Layers
	if layers <= 0 {
		return 0
	}
	cpu := cpuResidentLayers(cfg.GPULayers, layers)
	host := tm.WeightsBytes * int64(cpu) / int64(layers)
	if k := cpuMoeHostLayers(cfg, layers); k > cpu {
		host += tm.ExpertBytes * int64(k-cpu) / int64(layers)
	}
	return host
}

// cpuMoeHostLayers counts the leading blocks whose MoE experts --cpu-moe pins to
// the CPU: every block for the plain flag, the first N for --n-cpu-moe, none for
// a plan that emits neither. A config that sets both is read as the flag (all
// blocks), the conservative of the two - llama-server treats them as
// alternatives, so no plan the app writes has an expert placement this function
// would understate. hostWeightBytes is what turns this count into bytes, and it
// drops the leading blocks the host already runs.
//
// The experts of a dense model are 0 bytes (tuneModel.ExpertBytes), so this
// returns a count that prices nothing there; only a MoE load pays it.
func cpuMoeHostLayers(cfg ModelConfig, layers int) int {
	if cfg.CPUMoe {
		return layers
	}
	if cfg.NCpuMoe > 0 {
		if cfg.NCpuMoe > layers {
			return layers
		}
		return cfg.NCpuMoe
	}
	return 0
}

// cpuResidentLayers counts the blocks a gpu-layers setting leaves on the host. It
// is the dense half of the second copy a Mac load pays (hostWeightBytes prices it
// alongside the experts --cpu-moe pins to the host). Measured with phys_footprint
// (the dirty, non-reclaimable half) on the M2 at ctx 8192 and ubatch 512, every
// CPU-resident layer adds its own weight bytes again - the 36-layer 4B Q4_K_M
// (2381 MiB of weights, 66.1 MiB per layer) grew its dirty memory by 608 MiB with
// 9 layers on the CPU, 1092 with 18 and 2401 with all 36, and the 28-layer 0.6B
// (378 MiB, 13.5 per layer) by 134 with 14 and 371 with all 28. The mapped file
// pages stay swappable and unchanged (~2400 MiB), so this is llama.cpp holding the
// host-side tensors in its own buffers, not the file cache.
//
// "" / "auto" / "all" leave nothing on the host (the first two emit no -ngl flag,
// whose server-side default here is full offload). A number counts what it does
// not reach; an unreadable setting is priced as the worst case, and a model whose
// header could not be read arrives with buildTuneModel's 32-layer fallback rather
// than as "unknown, assume Metal".
func cpuResidentLayers(gpuLayers string, layers int) int {
	switch gpuLayers {
	case "", "auto", "all":
		return 0
	}
	n, err := strconv.Atoi(gpuLayers)
	if err != nil || n < 0 {
		return layers
	}
	if n >= layers {
		return 0
	}
	return layers - n
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
// and the memory that load prices (modelLoadPricing).
type plannedLoad struct {
	model ModelInfo
	cfg   ModelConfig
	bytes int64
	// pinned is the share of bytes inside the GPU working set: equal to bytes
	// everywhere but a Mac with a partial offload (modelLoadPricing).
	pinned int64
}

// residentSet is what router mode may hold in memory at the same time.
type residentSet struct {
	// loads are the entries that can be resident together, heaviest first,
	// capped at the server's max-loaded-models count.
	loads []plannedLoad
	// total is the sum of loads: the figure a session can drive the machine to.
	total int64
	// pinned is the same sum taken in the GPU's share of it - the bill the Metal
	// working set has to pay, which is where the pool ceiling is checked.
	pinned int64
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
		total, pinned := modelLoadPricing(m, c)
		loads = append(loads, plannedLoad{model: m, cfg: c, bytes: total, pinned: pinned})
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
		set.pinned += l.pinned
	}
	return set
}

// probeServingGPU is the serving-GPU seam for the router-mode over-commit
// warning (servingGPUPlan in production) so tests can drive the host-RAM gate
// without a GPU.
var probeServingGPU = servingGPUPlan

// probeMetalPoolMB is the GPU working-set seam for the launch-time memory
// decisions (metalWiredPoolMB in production) so tests can drive the pool clamp
// without a Metal device.
var probeMetalPoolMB = metalWiredPoolMB

// metalPoolBytes is the ceiling Metal may pin bytes into for a load served from
// Apple silicon, 0 when nothing bounds the load that way.
//
// The platform has to say so as well as the GPU snapshot, for the same reason
// loadRuntimeOverheadBytes checks both: only darwin + apple is the shape where
// the offloaded weights and the cache live in the device's working set, and a
// snapshot left over from another platform must never clamp a phone's or a Linux
// box's budget. A probe that cannot answer (0) leaves the host's free memory as
// the only ceiling, which is what it always was.
func metalPoolBytes(serving tuneTarget) int64 {
	if platformGOOS != "darwin" || serving.Vendor != vendorApple {
		return 0
	}
	return int64(probeMetalPoolMB()) * (1 << 20)
}

// memoryBudgetBytes is the ceiling a plan-shaping decision may size a context
// against: the host's free memory clamped by the Metal working set when that is
// the tighter of the two.
//
// This is the clamp the auto-tuner has applied to its own plans all along
// (tuneHardware's MetalPoolMB branch), and the reason the launch path needed it
// too is that the two figures do not track each other: on a 24 GiB M2 with
// freshly freed memory the pool is 18,186 MiB while the RAM budget is 18,360 MiB,
// so a plan can be priced inside the memory the app checks and still be over the
// device. Measured on that machine with a 378 MiB Qwen3-0.6B Q4_K_M at full
// offload, bracketing the context: 155,648 tokens load and generate, 159,744
// tokens loads without an error, passes /health, then fails every command buffer
// with kIOGPUCommandBufferCallbackErrorOutOfMemory and answers every later
// request with HTTP 500 "Compute error" until the model is reloaded.
//
// The figure is the whole bill only where the caller prices a full offload (the
// default, and the shape the measurement above ran in); see modelLoadPricing for
// the share a partial offload puts on the device, which is what the over-commit
// warning compares to the pool.
//
// An unknown availBytes stays unknown: a failed RAM probe says nothing about the
// machine, and the pool figure alone must not turn a silent start into a plan.
func memoryBudgetBytes(availBytes, poolBytes int64) int64 {
	if availBytes <= 0 {
		return availBytes
	}
	if poolBytes > 0 && poolBytes < availBytes {
		return poolBytes
	}
	return availBytes
}

// memoryLimitFor decides which ceiling a launch-time over-commit check should
// report for a priced library, and which bill to quote against it.
//
// It is not the same question as memoryBudgetBytes'. Clamping one budget to the
// tighter of the two ceilings would hide the case where the library passes the
// pool and busts RAM (or the reverse) by comparing the *other* ceiling's bill
// against it - here each ceiling is checked against the bill that ceiling actually
// pays: total against free RAM, the GPU-resident share against the pool. When RAM
// is over-committed that is what gets reported, because it is the larger problem
// and the sentence describing it is true; the pool is reported only when it is the
// limit the machine ran into first, which is the state that produces no failure
// anywhere else (memoryBudgetBytes' measurements).
//
// Returns (0, 0, false) when nothing is exceeded. A failed RAM probe (0) silences
// the RAM half only: the pool figure is a device property, not a free-memory
// reading, so an over-pool library is still worth saying.
func memoryLimitFor(availBytes, poolBytes, totalBytes, pinnedBytes int64) (limit, bill int64, poolBounded bool) {
	if poolBytes > 0 && pinnedBytes > poolBytes && (availBytes <= 0 || pinnedBytes <= availBytes) {
		return poolBytes, pinnedBytes, true
	}
	if availBytes > 0 && totalBytes > availBytes {
		return availBytes, totalBytes, false
	}
	return 0, 0, false
}

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
// "" when the library is empty or unreadable, the probe failed (limitBytes <= 0
// proves nothing), or the priced loads fit. suggestCtx is fittingCtxRung's
// answer for the heaviest model (>0 = the tier to move to, 0 = no tier on the
// ladder helps), computed by the caller so this stays a pure formatter.
//
// resident is how many loads totalBytes prices (planResidentSet's set size) and
// heaviestBytes is that one model's own share of them. At resident == 1 the two
// are the same load and the sentence stays the single-load one; above 1 it says
// how many can be resident and names the cap as the other lever, because the
// number the user can actually type is what makes the line actionable.
//
// limitBytes is the budget memoryBudgetBytes resolved and poolBounded says which
// limit it is, because over each one the failure looks different: past free RAM
// the machine swaps and crawls, past the GPU working set the load reports
// success and every request after it answers "Compute error" until the service
// restarts. Quoting the RAM figure for the second state would name the wrong
// ceiling and describe the wrong symptom - it is the machine's *other* number
// that ran out.
func overcommitWarning(name string, resident int, totalBytes, heaviestBytes, limitBytes int64, poolBounded bool, suggestCtx int) string {
	if name == "" || limitBytes <= 0 || totalBytes <= limitBytes {
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
	limit := fmt.Sprintf(tr("本机当前只有 %s 空闲内存。", "Only %s of memory is free right now. "), formatBytes(limitBytes))
	spill := tr("服务仍会启动、其他模型照常可用，但真去加载时内存放不下会开始交换：请求可能长时间没有响应、整台机器变卡。",
		"The service still starts and every other model stays usable, but loading it will overflow into swap: requests may hang for a long time and the whole machine slows down. ")
	if poolBounded {
		limit = fmt.Sprintf(tr("而这台机器 GPU 的工作集上限只有 %s：Metal 能钉住的量比空闲内存还少。",
			"This machine's GPU working set is capped at %s, which is less than the memory that is free. "), formatBytes(limitBytes))
		spill = tr("超过这个上限，加载照样报「成功」、健康检查也过，之后每个请求却只返回 Compute error，不重载模型就不会恢复。",
			"Over that ceiling a load still reports success and passes /health, and every request after it answers Compute error until the model is reloaded. ")
	}
	if resident > 1 {
		return fmt.Sprintf(tr("最多 %d 个模型会同时驻留，合计要占约 %s（每个模型各自的权重 + 上下文缓存 + 运行开销）；其中 %s 最重，它一项就要占约 %s。%s%s%s",
			"up to %d models can stay resident at once, which costs about %s in total (each model's own weights + context cache + runtime overhead); %s is the heaviest and takes about %s by itself. %s%s%s"),
			resident, formatBytes(totalBytes), name, formatBytes(heaviestBytes), limit, spill, advice)
	}
	return fmt.Sprintf(tr("模型 %s 这次要占约 %s（权重 + 上下文缓存 + 运行开销）。%s%s%s",
		"model %s needs about %s for this load (weights + context cache + runtime overhead). %s%s%s"),
		name, formatBytes(totalBytes), limit, spill, advice)
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
// memory the machine may actually spend (memoryBudgetBytes - free RAM, or the
// Metal working set where that is the tighter ceiling; an over-pool entry does
// not fail to load, it wedges the server) - never above tuneCtxMax, the ceiling
// every tuned plan is sized against, so an untuned 256k model cannot end up
// holding a bigger window than the tuned one would - and otherwise takes the
// largest rung of the tuner's ladder that fits, down to the ladder's floor.
// Unreadable metadata or a failed memory probe land on the settings page default,
// the same fallback the rest of this file uses.
func routerUntunedCtx(m ModelInfo) int {
	fallback := defaultModelConfig()
	mm, ok := readGGUFModelMetrics(m.Path)
	if !ok || mm.ContextLength <= 0 {
		return fallback.CtxSize
	}
	avail := memoryBudgetBytes(availableMemoryBytes(), metalPoolBytes(probeServingGPU()))
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
