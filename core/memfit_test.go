package core

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func pinProbeFreeMemoryGB(t *testing.T, gb float64) {
	t.Helper()
	orig := probeFreeMemoryGB
	probeFreeMemoryGB = func() float64 { return gb }
	t.Cleanup(func() { probeFreeMemoryGB = orig })
}

// pinProbeServingGPU pins the serving-GPU seam so the over-commit warning's
// host-RAM gate is driven without a GPU probe.
func pinProbeServingGPU(t *testing.T, target tuneTarget) {
	t.Helper()
	orig := probeServingGPU
	probeServingGPU = func() tuneTarget { return target }
	t.Cleanup(func() { probeServingGPU = orig })
}

// pinCPUSHapedReserve charges every load the 1280 MiB-at-512 CPU-only reserve.
// loadRuntimeOverheadBytes picks the figure from the platform plus the serving
// GPU probe, so a test that documents CPU arithmetic would otherwise get the
// Metal figure on an Apple Silicon machine and the Metal one nowhere else.
func pinCPUSHapedReserve(t *testing.T) {
	t.Helper()
	withPlatformGOOS(t, "linux")
	pinProbeServingGPU(t, tuneTarget{Vendor: vendorNone})
}

// pinMetalShapedReserve is the Apple Silicon twin: a Metal-serving load charges
// the 320 MiB-at-512 floor instead. Every plan these tests price is the tuner's
// full offload (or an unset gpu-layers), so no host-resident weight copy is
// added on top of the floor - see TestLoadRuntimeOverheadBytesShape for that term.
func pinMetalShapedReserve(t *testing.T) {
	t.Helper()
	withPlatformGOOS(t, "darwin")
	pinProbeServingGPU(t, tuneTarget{Vendor: vendorApple})
}

// TestAvailableMemoryBytes verifies the probe seam plumbing: the GiB value is
// converted to bytes, and a probe reporting nothing (0) or a negative value
// yields 0 so callers treat it as "unknown".
func TestAvailableMemoryBytes(t *testing.T) {
	for _, tt := range []struct {
		gb   float64
		want int64
	}{
		{2, 2 << 30},
		{1.5, int64(1.5 * float64(1<<30))},
		{0, 0},
		{-1, 0},
	} {
		pinProbeFreeMemoryGB(t, tt.gb)
		if got := availableMemoryBytes(); got != tt.want {
			t.Errorf("availableMemoryBytes(%v GiB) = %d, want %d", tt.gb, got, tt.want)
		}
	}
}

// TestMemoryFitError walks the fit boundary: a load that fits (and one exactly
// at the limit) is accepted, a probe failure never blocks, and a load that
// cannot fit produces an error carrying both measured figures plus, when the
// caller has one, the context tier that would fit.
func TestMemoryFitError(t *testing.T) {
	gib := int64(1 << 30)
	tests := []struct {
		name     string
		need     int64
		avail    int64
		suggest  int
		wantErr  bool
		wantTier bool // the message must quote the suggested tier
	}{
		{"fits", 3 * gib, 5 * gib, 0, false, false},
		{"exactly at the limit fits", 4 * gib, 4 * gib, 0, false, false},
		{"one byte over is refused", 4*gib + 1, 4 * gib, 0, true, false},
		{"the fitting tier is quoted", 4*gib + 1, 4 * gib, 16384, true, true},
		{"unknown availability never blocks", 20 * gib, 0, 16384, false, false},
		{"failed probe (negative) never blocks", 20 * gib, -1, 16384, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := memoryFitError("qwen-4b", tt.need, tt.avail, tt.suggest)
			if (err != nil) != tt.wantErr {
				t.Fatalf("memoryFitError(%d, %d) err = %v, wantErr %v", tt.need, tt.avail, err, tt.wantErr)
			}
			if err == nil {
				return
			}
			// The message is the only thing the user gets: it must name the
			// model and both numbers, whatever UI language tr() resolved to.
			msg := err.Error()
			for _, want := range []string{"qwen-4b", formatBytes(tt.need), formatBytes(tt.avail)} {
				if !strings.Contains(msg, want) {
					t.Errorf("error %q missing %q", msg, want)
				}
			}
			// The tier is the actionable half. Asserted by its digits because
			// the surrounding sentence is tr()-resolved and may be English.
			if tt.wantTier && !strings.Contains(msg, strconv.Itoa(tt.suggest)) {
				t.Errorf("error %q does not quote the fitting tier %d", msg, tt.suggest)
			}
			if !tt.wantTier && tt.suggest > 0 && strings.Contains(msg, strconv.Itoa(tt.suggest)) {
				t.Errorf("error %q quotes a tier the caller did not resolve", msg)
			}
		})
	}
}

// writeMemfitGGUF writes a qwen35 hybrid-attention fixture: 36 blocks with KV
// on every 3rd layer (12 cache layers) and 8 KV heads at 128-dim key/value, so
// one token costs 8*(128+128)*2*12 = 49152 bytes of f16 cache.
func writeMemfitGGUF(t *testing.T, dir string) string {
	t.Helper()
	return writeTempGGUF(t, dir, "readable.gguf", buildGGUF(3,
		strKV("general.architecture", "qwen35"),
		u64KV("qwen35.block_count", 36),
		u64KV("qwen35.context_length", 262144),
		u32KV("qwen35.attention.head_count", 32),
		u32KV("qwen35.attention.head_count_kv", 8),
		u32KV("qwen35.attention.key_length", 128),
		u32KV("qwen35.attention.value_length", 128),
		u32KV("qwen35.full_attention_interval", 3),
	))
}

// writeMemfitSwaGGUF writes a gemma-3-1b-shaped sliding-window fixture: 26
// layers, one KV head of 256-dim key and value (1024 B of f16 cache per layer
// per token) and a 512-token sliding window, i.e. the geometry the measured
// checkpoints were priced against upstream b11720.
func writeMemfitSwaGGUF(t *testing.T, dir string) string {
	t.Helper()
	return writeTempGGUF(t, dir, "swa.gguf", buildGGUF(3,
		strKV("general.architecture", "gemma3"),
		u64KV("gemma3.block_count", 26),
		u64KV("gemma3.context_length", 32768),
		u32KV("gemma3.attention.head_count", 4),
		u32KV("gemma3.attention.head_count_kv", 1),
		u32KV("gemma3.attention.key_length", 256),
		u32KV("gemma3.attention.value_length", 256),
		u32KV("gemma3.attention.sliding_window", 512),
	))
}

// TestModelLoadBytes pins the four terms of the planned footprint and how the KV
// and checkpoint terms react to the model metadata and the configured context.
func TestModelLoadBytes(t *testing.T) {
	pinCPUSHapedReserve(t)
	const (
		mib     = int64(1 << 20)
		gib     = int64(1 << 30)
		overhd  = int64(tuneRuntimeOverheadMB) * mib // priced at the default ubatch
		fourGib = 4 * gib
		// Unreadable metadata falls back to the tuner's conservative geometry:
		// 32 layers x 1024 B/token = 32 KiB per token.
		fallbackKV4096 = 32 * 1024 * 4096
		hybridKV4096   = 49152 * 4096
		// The sliding-window fixture's own geometry: 26 layers x 1024 B/token
		// of f16 cache, 512-token window, so one checkpoint copy is priced at
		// 26 x 1024 x 512 = 13.0 MiB (measured: 11.0 MiB for the 22 window
		// layers of the same model).
		swaPerTok      = int64(26 * 1024)
		swaCheckpoints = int64(ctxCheckpointCap) * swaPerTok * 512
	)
	dir := t.TempDir()
	readable := writeMemfitGGUF(t, dir)
	swa := writeMemfitSwaGGUF(t, dir)
	unreadable := writeTempGGUF(t, dir, "junk.gguf", make([]byte, 64))
	mmprojDir := t.TempDir()
	mmprojModel := writeTempGGUF(t, mmprojDir, "vision.gguf", make([]byte, 64))
	const mmprojSize = int64(4096)
	if err := os.WriteFile(filepath.Join(mmprojDir, "mmproj-vision.gguf"), ggufFileBytes(int(mmprojSize)), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		m    ModelInfo
		cfg  ModelConfig
		want int64
	}{
		{
			// No context configured: weights and the runtime overhead only.
			name: "weights + overhead, ctx unset",
			m:    ModelInfo{Name: "m", Path: unreadable, SizeBytes: fourGib},
			cfg:  ModelConfig{},
			want: fourGib + overhd,
		},
		{
			// A model whose metadata cannot be read is still charged the
			// tuner's fallback KV, so the guard and the tuner agree.
			name: "fallback KV charged for unreadable metadata",
			m:    ModelInfo{Name: "m", Path: unreadable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096},
			want: fourGib + fallbackKV4096 + overhd,
		},
		{
			// Real geometry, hybrid attention: only the 12 KV-bearing layers
			// are charged, not all 36.
			name: "f16 KV over the KV-bearing layers",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096},
			want: fourGib + hybridKV4096 + overhd,
		},
		{
			// The q8_0 pair the auto-tuner keeps is priced exactly as the tuner
			// prices its context ladder (17/32 of f16), so the guard can never
			// refuse the plan the tuner just selected.
			name: "pinned q8_0 pair priced at the cache ratio",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096, CacheTypeK: "q8_0", CacheTypeV: "q8_0"},
			want: fourGib + hybridKV4096*17/32 + overhd,
		},
		{
			// Half a pair is two different buffers, and the guard prices each
			// one: K at q8_0 plus V at f16 measured 0.7656 of an all-f16
			// cache on this backend, which is exactly the mean of 17/32 and
			// 1 (49/64). The mirrored selection (V quantized, K f16) measured
			// the same 0.7656, so the halves are interchangeable here.
			name: "one-sided cache pin charges the mean of its halves",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096, CacheTypeK: "q8_0"},
			want: fourGib + hybridKV4096*49/64 + overhd,
		},
		{
			// q4_0 measured 0.2816 of f16 (32,307 B/token against the model's
			// 114,746 B f16 geometry) on the same backend.
			name: "q4_0 pair charges its measured ratio",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096, CacheTypeK: "q4_0", CacheTypeV: "q4_0"},
			want: fourGib + hybridKV4096*2816/10000 + overhd,
		},
		{
			// A type outside llama.cpp's list (preset.go would refuse to write
			// it, but a hand-edited config can carry anything) charges the
			// f16 upper bound rather than a guessed ratio.
			name: "unknown cache type stays at f16",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096, CacheTypeK: "made_up", CacheTypeV: "made_up"},
			want: fourGib + hybridKV4096 + overhd,
		},
		{
			name: "same-directory mmproj counts as weights",
			m:    ModelInfo{Name: "m", Path: mmprojModel, SizeBytes: fourGib, HasMMProj: true},
			cfg:  ModelConfig{},
			want: fourGib + mmprojSize + overhd,
		},
		{
			// Raising --ubatch-size to 2048 costs 4x the activation buffers
			// (measured: the idle-to-first-batch step is linear in ubatch), so
			// the guard charges 4x the default-ubatch reserve.
			name: "runtime overhead priced at the configured ubatch",
			m:    ModelInfo{Name: "m", Path: unreadable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 4096, UBatchSize: 2048},
			want: fourGib + fallbackKV4096 + 4*overhd,
		},
		{
			// The fourth term: a sliding-window model also pays for the
			// checkpoint budget its plan runs with. At this context the
			// retention is twice the cache it copies (8 x 512 window tokens
			// against 2048 context tokens), which is the whole reason the
			// budget is capped and priced instead of left at upstream's 32.
			name: "checkpoint budget charged for a sliding-window model",
			m:    ModelInfo{Name: "m", Path: swa, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 2048},
			want: fourGib + swaPerTok*2048 + swaCheckpoints + overhd,
		},
		{
			// The pin the settings page writes (and every Android and
			// Windows full-offload plan gets) turns the term back off, and
			// the preset emits the matching ctx-checkpoints = 0.
			name: "pinned-off plan charges no checkpoints",
			m:    ModelInfo{Name: "m", Path: swa, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 2048, CtxCheckpointsOff: true},
			want: fourGib + swaPerTok*2048 + overhd,
		},
		{
			// A plain-attention model creates no checkpoints at all (measured:
			// Qwen3-4B, one 22,321-token prompt, zero checkpoint lines), so the
			// term must not appear for it even though it runs at the cap.
			name: "no checkpoint term without a sliding window",
			m:    ModelInfo{Name: "m", Path: readable, SizeBytes: fourGib},
			cfg:  ModelConfig{CtxSize: 2048},
			want: fourGib + 49152*2048 + overhd,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := modelLoadBytes(tt.m, tt.cfg); got != tt.want {
				t.Errorf("modelLoadBytes = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestLoadRuntimeOverheadBytesShape pins how the reserve is priced per shape:
// the Metal floor alone for the load that runs every layer on Metal, plus one
// more copy of the weights of the layers the host still computes, and the
// CPU-only measurement on any other platform - a Mac-shaped reserve must never
// be reachable from linux, android or windows, and a Mac with a CUDA card in the
// probe pays the CPU figure too. The floor and the copy are both measured on the
// M2 (see loadRuntimeOverheadBytes / cpuResidentLayers); the 3600-MiB-over-36
// layers fixture below puts them at 100 MiB per host-resident block.
func TestLoadRuntimeOverheadBytesShape(t *testing.T) {
	const mib = int64(1 << 20)
	cases := []struct {
		name      string
		goos      string
		vendor    string
		gpuLayers string
		layers    int
		weightsMB int64
		ubatch    int
		wantMB    int64
	}{
		{"metal auto", "darwin", vendorApple, "auto", 36, 3600, 512, 320},
		{"metal all", "darwin", vendorApple, "all", 36, 3600, 512, 320},
		// An unset value emits no -ngl flag, and llama-server's own default on
		// Apple silicon is full offload (measured: the RSS grows by the f16 KV
		// geometry with no flag at all), so it is the floor alone too.
		{"metal default", "darwin", vendorApple, "", 36, 3600, 512, 320},
		{"metal exact layer count", "darwin", vendorApple, "36", 36, 3600, 512, 320},
		{"metal over-count", "darwin", vendorApple, "99", 36, 3600, 512, 320},
		// Measured dirty memory over the offloaded run: 9 of 36 blocks left on
		// the host cost +608 MiB and 18 of them +1092 MiB on the 4B Q4_K_M
		// (66.1 MiB of weights per block), i.e. the weight copy, not a flat
		// reserve. The 1 GiB the CPU-only figure omitted is what made -ngl 0
		// under-price by three quarters of the model.
		{"one block on the host", "darwin", vendorApple, "35", 36, 3600, 512, 420},
		{"nine blocks on the host", "darwin", vendorApple, "27", 36, 3600, 512, 1220},
		{"eighteen blocks on the host", "darwin", vendorApple, "18", 36, 3600, 512, 2120},
		{"metal explicit cpu", "darwin", vendorApple, "0", 36, 3600, 512, 3920},
		// A model whose header could not be read arrives with the tuner's
		// 32-layer fallback, so "unknown geometry" prices the fallback split
		// rather than the floor.
		{"number, 32-layer fallback", "darwin", vendorApple, "20", 32, 3200, 512, 1520},
		// An unparseable or negative setting is the worst case: every block on
		// the host.
		{"garbage setting", "darwin", vendorApple, "half", 36, 3600, 512, 3920},
		{"negative setting", "darwin", vendorApple, "-1", 36, 3600, 512, 3920},
		// The grammar the frontend mirror parses with is Atoi's, not Number()'s:
		// a signed integer counts, a padded or fractional one is the worst case
		// (preset.go refuses such a value before the server ever sees it).
		{"signed full offload", "darwin", vendorApple, "+36", 36, 3600, 512, 320},
		{"padded number", "darwin", vendorApple, " 36 ", 36, 3600, 512, 3920},
		{"fractional number", "darwin", vendorApple, "12.5", 36, 3600, 512, 3920},
		{"mac with a cuda card", "darwin", "nvidia", "all", 36, 3600, 512, 1280},
		{"linux probing apple", "linux", vendorApple, "0", 36, 3600, 512, 1280},
		{"android probing apple", "android", vendorApple, "all", 36, 3600, 512, 1280},
		{"windows cpu only", "windows", vendorNone, "0", 36, 3600, 512, 1280},
		// The ubatch scaling prices the floor on both shapes; the weight copy is
		// a property of the model, not of the batch size.
		{"metal at 1024", "darwin", vendorApple, "all", 36, 3600, 1024, 640},
		{"metal split at 1024", "darwin", vendorApple, "0", 36, 3600, 1024, 4240},
		{"metal ubatch unset", "darwin", vendorApple, "all", 36, 3600, 0, 320},
		{"cpu at 2048", "linux", vendorNone, "0", 36, 3600, 2048, 5120},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			orig := platformGOOS
			platformGOOS = tt.goos
			t.Cleanup(func() { platformGOOS = orig })
			tm := tuneModel{Layers: tt.layers, WeightsBytes: tt.weightsMB * mib}
			got := loadRuntimeOverheadBytes(ModelConfig{GPULayers: tt.gpuLayers, UBatchSize: tt.ubatch},
				tuneTarget{Vendor: tt.vendor}, tm)
			if got != tt.wantMB*mib {
				t.Errorf("loadRuntimeOverheadBytes(%s, %s, gpuLayers %q, %d layers of %d MiB, ubatch %d) = %d MiB, want %d",
					tt.goos, tt.vendor, tt.gpuLayers, tt.layers, tt.weightsMB, tt.ubatch, got/mib, tt.wantMB)
			}
		})
	}
}

// TestMetalReserveBuysContext is the user-visible half of the Metal shape: the
// same 4 GiB model at ctx 8192 is 960 MiB cheaper to plan on a Mac when the plan
// offloads every one of the fixture's 36 layers - and the reserve grows again
// with each block left on the host, past the CPU-only figure.
func TestMetalReserveBuysContext(t *testing.T) {
	dir := t.TempDir()
	path := writeMemfitGGUF(t, dir)
	m := ModelInfo{Name: "metal-model", Path: path, SizeBytes: 4 * (1 << 30)}
	cfg := ModelConfig{CtxSize: 8192, GPULayers: "all", UBatchSize: 512}

	orig := platformGOOS
	platformGOOS = "darwin"
	t.Cleanup(func() { platformGOOS = orig })

	pinProbeServingGPU(t, tuneTarget{Vendor: vendorNone})
	cpu := modelLoadBytes(m, cfg)
	pinProbeServingGPU(t, tuneTarget{Vendor: vendorApple})
	metal := modelLoadBytes(m, cfg)

	if diff := cpu - metal; diff != (tuneRuntimeOverheadMB-tuneMetalRuntimeOverheadMB)*(1<<20) {
		t.Errorf("metal load is %d MiB cheaper, want %d (the reserve difference at this ubatch)",
			diff/(1<<20), (tuneRuntimeOverheadMB - tuneMetalRuntimeOverheadMB))
	}
	// Each host-resident block is charged again as weight bytes: this fixture's
	// 4 GiB over 36 layers is a quarter of the model at -ngl 27, a half at -ngl
	// 18 and the whole model at -ngl 0, which is what the phys_footprint sweep
	// measured. The flat CPU figure was not just wrong for these plans, it was
	// too small - the -ngl 0 load holds a second copy of the file, which is
	// 3136 MiB more than the CPU-only reserve admits.
	for _, tt := range []struct {
		gpuLayers string
		wantMB    int64
	}{
		{"27", 1024},
		{"18", 2048},
		{"0", 4096},
	} {
		split := modelLoadBytes(m, ModelConfig{CtxSize: 8192, GPULayers: tt.gpuLayers, UBatchSize: 512})
		if want := metal + tt.wantMB*(1<<20); split != want {
			t.Errorf("-ngl %s load = %d MiB, want %d (the Metal floor plus %d MiB of host weights)",
				tt.gpuLayers, split/(1<<20), want/(1<<20), tt.wantMB)
		}
		if tt.gpuLayers == "0" && split <= cpu {
			t.Errorf("-ngl 0 load = %d MiB, must exceed the CPU-only figure %d",
				split/(1<<20), cpu/(1<<20))
		}
	}
	// Covering the block count exactly is the Metal floor alone, over-counting too.
	for _, gpuLayers := range []string{"36", "99", "auto", ""} {
		offloaded := modelLoadBytes(m, ModelConfig{CtxSize: 8192, GPULayers: gpuLayers, UBatchSize: 512})
		if offloaded != metal {
			t.Errorf("-ngl %q load = %d MiB, want the Metal-shaped %d",
				gpuLayers, offloaded/(1<<20), metal/(1<<20))
		}
	}
	// The saved reserve is free memory as far as the budget is concerned, so the
	// router retains more prompt cache with the same model loaded.
	const avail = int64(6 << 30)
	cpuCache := promptCacheRAMMiB(8192, cpu, avail)
	metalCache := promptCacheRAMMiB(8192, metal, avail)
	if metalCache <= cpuCache {
		t.Errorf("prompt-cache budget = metal %d MiB, cpu %d MiB: the cheaper load must leave more free", metalCache, cpuCache)
	}
}

// TestCtxCheckpointBytes pins the checkpoint term's formula and its switches on
// the measured gemma-3-1b geometry (26 layers, 1 KV head of 256-dim key and
// value = 1024 B of f16 cache per layer per token, 512-token window): each
// checkpoint the server logged for that model was 11.006 MiB, matching 22 window
// layers, while this term prices all 26 (13.0 MiB) — the 18% over-charge a
// start-time budget wants in its error direction.
func TestCtxCheckpointBytes(t *testing.T) {
	const (
		perTok   = int64(26 * 1024)
		oneCkpt  = perTok * 512
		capBytes = int64(ctxCheckpointCap) * oneCkpt
	)
	swa := tuneModel{Layers: 26, KVBytesPerTokPerLayerF16: 1024, SwaWindow: 512}
	tests := []struct {
		name string
		tm   tuneModel
		cfg  ModelConfig
		goos string
		want int64
	}{
		{"the cap times one window-sized copy", swa, ModelConfig{CtxSize: 4096}, "darwin", capBytes},
		{
			// The q8_0 pair shrinks the copy like it shrinks the cache: the
			// same checkpoints measured 5.85 MiB against 11.0 MiB at f16,
			// i.e. the 0.53125 ratio again.
			"quantized cache prices the copies at its ratio", swa,
			ModelConfig{CtxSize: 4096, CacheTypeK: "q8_0", CacheTypeV: "q8_0"}, "darwin",
			int64(ctxCheckpointCap) * perTok * 17 / 32 * 512,
		},
		{
			// A checkpoint copies the window, not the context: below the
			// window the smaller of the two is what can be held.
			"window clamped by a smaller context", swa, ModelConfig{CtxSize: 256}, "darwin",
			int64(ctxCheckpointCap) * perTok * 256,
		},
		{"no sliding window copies nothing", tuneModel{Layers: 26, KVBytesPerTokPerLayerF16: 1024}, ModelConfig{CtxSize: 4096}, "darwin", 0},
		{"pinned off", swa, ModelConfig{CtxSize: 4096, CtxCheckpointsOff: true}, "darwin", 0},
		{
			// Android is a platform rule, not a config field: it holds even
			// for a stored config that never mentions checkpoints.
			"android runs no checkpoints", swa, ModelConfig{CtxSize: 4096}, "android", 0,
		},
		{"no context configured yet", swa, ModelConfig{}, "darwin", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withPlatformGOOS(t, tt.goos)
			if got := ctxCheckpointBytes(tt.tm, tt.cfg); got != tt.want {
				t.Errorf("ctxCheckpointBytes = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestPromptCacheRAMMiB walks the prompt-cache bound: a budget the free memory
// covers is left alone, an oversized one is cut to the share of the headroom, a
// load with no headroom gets no cache, and neither an unknown probe nor an
// already-off cache is touched.
func TestPromptCacheRAMMiB(t *testing.T) {
	const mib = int64(1 << 20)
	tests := []struct {
		name  string
		req   int
		need  int64
		avail int64
		want  int
	}{
		{"a budget the headroom covers stays as set", 512, 2048 * mib, 8192 * mib, 512},
		{"an oversized budget is bounded to the headroom share", 8192, 5504 * mib, 6144 * mib, 320},
		{"a load that uses all the memory gets no cache", 8192, 6144 * mib, 6144 * mib, 0},
		{"an unknown probe invents no bound", 8192, 6144 * mib, 0, 8192},
		{"an off cache stays off", 0, 1024 * mib, 8192 * mib, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := promptCacheRAMMiB(tt.req, tt.need, tt.avail); got != tt.want {
				t.Errorf("promptCacheRAMMiB(%d, need %d, avail %d) = %d, want %d",
					tt.req, tt.need, tt.avail, got, tt.want)
			}
		})
	}
}

// TestTunedQ8PlanPassesTheMemoryGuard closes the loop between the two places
// that price a load: a RAM-bounded plan that sized its context ladder at the
// pinned q8_0 cache must survive the start-time guard, or the app would refuse
// the configuration it just wrote (on a phone, that is a start that never
// happens plus a message telling the user to do what they already did).
//
// Fixture: 4 GiB weights, 48 KiB of f16 KV per token, 1.25 GiB reserve at the
// default ubatch, 10 GiB free of 12 GiB total (a phone-class memory picture).
// The plan's budget is 8.5 GiB, so the q8_0 ladder reaches 131072 at
// 4 + 3.1875 + 1.25 = 8.4375 GiB; that same window priced at f16 needs 11.25
// GiB, which is more than the device has.
func TestTunedQ8PlanPassesTheMemoryGuard(t *testing.T) {
	// The fixture is a phone-class memory picture (Android platform, 12 GiB), so
	// the probe has to agree with it: without this pin the same plan priced on an
	// Apple Silicon dev machine would take the Metal reserve and refuse itself.
	pinCPUSHapedReserve(t)
	dir := t.TempDir()
	path := writeMemfitGGUF(t, dir)
	const fourGib = int64(4) * (1 << 30)
	m := ModelInfo{Name: "phone-model", Path: path, SizeBytes: fourGib}

	metrics, ok := readGGUFModelMetrics(path)
	if !ok {
		t.Fatal("fixture GGUF is not readable")
	}
	tm := buildTuneModel(metrics, ok, tuneWeightsBytes(m, ModelConfig{}))
	tm.PinnedQ8KV = true
	hw := tuneHardware{
		GPUVendor:       vendorNone,
		RAMTotalGB:      12,
		RAMFreeGB:       10,
		PhysicalCores:   8,
		AndroidPlatform: true,
	}
	cfg := tuneModelConfig(hw, tm)
	if cfg.CtxSize != 131072 || cfg.CacheTypeK != "q8_0" || cfg.CacheTypeV != "q8_0" {
		t.Fatalf("tuner plan = ctx %d cacheK=%q cacheV=%q, want 131072 with the q8_0 pair",
			cfg.CtxSize, cfg.CacheTypeK, cfg.CacheTypeV)
	}

	pinProbeFreeMemoryGB(t, 10)
	if err := modelLoadFitError(m, cfg); err != nil {
		t.Errorf("modelLoadFitError(tuned q8_0 plan) = %v, want nil: the guard must not refuse the plan the tuner emitted", err)
	}
	// The same window without the cache pair really is unaffordable, so the
	// accepted start above comes from the halved cache and not from slack.
	if err := modelLoadFitError(m, ModelConfig{CtxSize: cfg.CtxSize}); err == nil {
		t.Error("modelLoadFitError(f16-priced window) = nil, want a refusal: the fixture no longer straddles the boundary")
	}
}

// TestTunedUbatchPlanPassesTheMemoryGuard is the same loop closed for the other
// pricing input: a model whose user raised --ubatch-size must get a plan priced
// at that ubatch, or the guard refuses the configuration the tuner wrote.
//
// Same phone fixture (4 GiB weights, 48 KiB of f16 KV per token, q8_0 pinned, 10
// GiB free of 12): at the carried ubatch 2048 the runtime overhead is 5 GiB, so
// nothing but the smallest ladder rung is affordable and the load costs
// 4 + 5 GiB + the 2048-token cache. Repriced at the default 1.25 GiB reserve the
// plan would have reached 131072 — which at ubatch 2048 needs 12.19 GiB, more
// than the device has.
func TestTunedUbatchPlanPassesTheMemoryGuard(t *testing.T) {
	pinCPUSHapedReserve(t)
	dir := t.TempDir()
	path := writeMemfitGGUF(t, dir)
	m := ModelInfo{Name: "phone-model", Path: path, SizeBytes: 4 * (1 << 30)}

	metrics, ok := readGGUFModelMetrics(path)
	if !ok {
		t.Fatal("fixture GGUF is not readable")
	}
	tm := buildTuneModel(metrics, ok, tuneWeightsBytes(m, ModelConfig{}))
	tm.PinnedQ8KV = true
	tm.UBatchSize = 2048
	cfg := tuneModelConfig(tuneHardware{
		GPUVendor:       vendorNone,
		RAMTotalGB:      12,
		RAMFreeGB:       10,
		PhysicalCores:   8,
		AndroidPlatform: true,
	}, tm)
	if cfg.UBatchSize != 2048 || cfg.CtxSize != tuneCtxMin {
		t.Fatalf("tuner plan = ubatch %d ctx %d, want 2048 / %d: the 5 GiB overhead must be paid in context",
			cfg.UBatchSize, cfg.CtxSize, tuneCtxMin)
	}

	pinProbeFreeMemoryGB(t, 10)
	if err := modelLoadFitError(m, cfg); err != nil {
		t.Errorf("modelLoadFitError(ubatch-2048 plan) = %v, want nil: the guard must not refuse the plan the tuner emitted", err)
	}
	// The mismatch this closes: the same model tuned at the default ubatch keeps
	// its 131072 window, and adding the user's 2048 ubatch afterwards is a start
	// that gets refused.
	if err := modelLoadFitError(m, ModelConfig{CtxSize: 131072, CacheTypeK: "q8_0", CacheTypeV: "q8_0", UBatchSize: 2048}); err == nil {
		t.Error("modelLoadFitError(131072 window at ubatch 2048) = nil, want a refusal: the fixture no longer straddles the boundary")
	}
}

// TestTunedSwaPlanPricesCheckpointsInTheLadder closes the same loop for the
// checkpoint term: a sliding-window model's plan must budget the retention the
// start guard charges, or the tuner selects a rung the memory decision then
// refuses (on a desktop that is a warning naming a tier the app just chose).
//
// Metal-shaped fixture and probe (loadRuntimeOverheadBytes' 320 MiB reserve): 4
// GiB of weights, 26 KiB of f16 KV per token, a 512-token window and 6.1 GiB
// free of 8, which leaves the tuner 5,309 MiB after its margins. The 32768 rung
// costs 4,096 + 832 + 320 = 5,248 MiB without checkpoints — affordable — and
// 5,352 MiB once the 104 MiB budget is priced with it, so the plan drops to
// 16384 (4,096 + 416 + 320 + 104 = 4,936 MiB).
func TestTunedSwaPlanPricesCheckpointsInTheLadder(t *testing.T) {
	pinMetalShapedReserve(t)
	dir := t.TempDir()
	path := writeMemfitSwaGGUF(t, dir)
	m := ModelInfo{Name: "swa-model", Path: path, SizeBytes: 4 * (1 << 30)}

	metrics, ok := readGGUFModelMetrics(path)
	if !ok {
		t.Fatal("fixture GGUF is not readable")
	}
	tm := buildTuneModel(metrics, ok, tuneWeightsBytes(m, ModelConfig{}))
	if tm.SwaWindow != 512 {
		t.Fatalf("fixture SwaWindow = %d, want 512", tm.SwaWindow)
	}
	cfg := tuneModelConfig(tuneHardware{
		GPUVendor:     vendorApple,
		RAMTotalGB:    8,
		RAMFreeGB:     6.1,
		PhysicalCores: 8,
	}, tm)
	if cfg.CtxSize != 16384 {
		t.Fatalf("tuner plan ctx = %d, want 16384: the checkpoint budget belongs in the ladder price", cfg.CtxSize)
	}

	pinProbeFreeMemoryGB(t, 5)
	if err := modelLoadFitError(m, cfg); err != nil {
		t.Errorf("modelLoadFitError(tuned SWA plan) = %v, want nil: the guard must not refuse the plan the tuner emitted", err)
	}
	// The rung a checkpoint-blind ladder would have kept really is unaffordable
	// at this availability, so the plan above changed because of the term.
	if err := modelLoadFitError(m, ModelConfig{CtxSize: 32768}); err == nil {
		t.Error("modelLoadFitError(32768) = nil, want a refusal: the fixture no longer straddles the boundary")
	}
}

// TestModelLoadFitError verifies the scanned-model entry point against the
// probe: the overhead and the context are part of the decision, and a failed
// probe never blocks a start.
func TestModelLoadFitError(t *testing.T) {
	pinCPUSHapedReserve(t)
	const (
		gib        = int64(1 << 30)
		overheadGB = float64(tuneRuntimeOverheadMB) / 1024.0 // 1.25 GiB
		fallbackGB = float64(32*1024*4096) / float64(1<<30)  // 0.125 GiB at ctx 4096
	)
	m := ModelInfo{Name: "huge-model", Path: "/nonexistent.gguf", SizeBytes: 8 * gib}

	// 8 GiB of weights + 1.25 GiB of overhead fits in 10 GiB.
	pinProbeFreeMemoryGB(t, 10)
	if err := modelLoadFitError(m, ModelConfig{}); err != nil {
		t.Errorf("10 GiB available must accept an 8 GiB model, got %v", err)
	}

	// The weights alone would fit 9 GiB; the runtime overhead is what refuses.
	pinProbeFreeMemoryGB(t, 9)
	if err := modelLoadFitError(m, ModelConfig{}); err == nil {
		t.Error("9 GiB available must refuse 8 GiB of weights plus the runtime overhead")
	}

	// Between "weights + overhead" and "weights + overhead + KV": the same
	// model is accepted with no context and refused once one is configured.
	avail := 8 + overheadGB + fallbackGB/2
	pinProbeFreeMemoryGB(t, avail)
	if err := modelLoadFitError(m, ModelConfig{}); err != nil {
		t.Errorf("%v GiB available must accept the load without KV, got %v", avail, err)
	}
	if err := modelLoadFitError(m, ModelConfig{CtxSize: 4096}); err == nil {
		t.Errorf("%v GiB available must refuse once the KV cache is charged", avail)
	}

	pinProbeFreeMemoryGB(t, 0)
	if err := modelLoadFitError(m, ModelConfig{CtxSize: 4096}); err != nil {
		t.Errorf("failed probe must not block a start, got %v", err)
	}

	// The advice half through the real entry point, not just the formatter: the
	// availability above pays for the 2048 tier exactly (8 GiB weights + 1280 MiB
	// overhead + 2048 tokens x the 32 KiB/token fallback geometry = the same
	// 9.3125 GiB), so the refusal of the 4096 load has to name that rung rather
	// than leave the user with two numbers.
	pinProbeFreeMemoryGB(t, avail)
	err := modelLoadFitError(m, ModelConfig{CtxSize: 4096})
	if err == nil {
		t.Fatal("ctx 4096 must still be refused at this availability")
	}
	if want := strconv.Itoa(tuneCtxMin); !strings.Contains(err.Error(), want) {
		t.Errorf("refusal = %v, want it to name the %s tier the same pricing accepts", err, want)
	}
	// And the same for a far bigger plan: this machine pays for exactly one rung
	// (2048 costs 9,536 MiB, 4,096 costs 9,600 MiB), so the ceiling refusal names
	// that one rung too - never anything above the configured context's price.
	if err := modelLoadFitError(m, ModelConfig{CtxSize: tuneCtxMax}); err == nil {
		t.Fatal("the ladder ceiling must be refused at this availability")
	} else if want := strconv.Itoa(tuneCtxMin); !strings.Contains(err.Error(), want) {
		t.Errorf("ctx %d refusal = %v, want it to name the only affordable rung %s", tuneCtxMax, err, want)
	}
}

// TestDirectModelConfigFallsBackToAppDefaults pins the direct-mode config
// resolution: a stored config wins as-is, and a model the user never touched
// gets the app default rather than a zero value (which would leave
// --ctx-size unset and hand llama-server the model's trained context).
func TestDirectModelConfigFallsBackToAppDefaults(t *testing.T) {
	stored := ModelConfig{CtxSize: 32768, Threads: 4}
	cfgs := map[string]ModelConfig{
		"tuned":  stored,
		"legacy": {Threads: 4, BatchSize: 2048},
	}

	if got := directModelConfig(cfgs, "tuned"); got != stored {
		t.Errorf("stored config = %+v, want %+v", got, stored)
	}
	if got := directModelConfig(cfgs, "untouched"); got.CtxSize != defaultModelConfig().CtxSize {
		t.Errorf("untouched model CtxSize = %d, want the app default %d", got.CtxSize, defaultModelConfig().CtxSize)
	}
	// A stored entry that carries no context is legacy or hand-edited data, not
	// an instruction to leave --ctx-size unset: every other field of it still
	// wins, only the context is filled.
	want := ModelConfig{CtxSize: 4096, Threads: 4, BatchSize: 2048}
	if got := directModelConfig(cfgs, "legacy"); got != want {
		t.Errorf("contextless stored entry = %+v, want %+v", got, want)
	}
}

// TestContextlessStoredConfigIsGuardHonest is the pairing the phone needs: the
// config the start guard prices and the args the server actually gets must
// describe the same load. Priced with no context this fixture is 2,304 MiB
// (1,024 weights + 1,280 overhead) and gets accepted with 2,312 MiB free, while
// the server it spawns runs the trained 32,768-token window - 2,368 MiB of KV and
// weights - i.e. the load that gets killed is the one the check approved. Once
// the resolver fills the default context, the same device prices the real
// 2,312 MiB and refuses at one MiB less.
func TestContextlessStoredConfigIsGuardHonest(t *testing.T) {
	pinCPUSHapedReserve(t)
	dir := t.TempDir()
	path := geomGGUF(t, dir, "geom.gguf")
	m := ModelInfo{Name: "legacy-model", Path: path, SizeBytes: 1 << 30}
	cfgs := map[string]ModelConfig{"legacy-model": {Threads: 4}}

	pinProbeFreeMemoryGB(t, gibFromMiB(2312)) // 1024 weights + 1280 overhead + 8 KV (4096 x 2 KiB)
	resolved := directModelConfig(cfgs, m.Name)
	if err := modelLoadFitError(m, resolved); err != nil {
		t.Errorf("the price the resolver describes = %v, want it accepted at exactly the affordable figure", err)
	}
	// One MiB short and the same load is refused, because the resolver's context
	// is what the guard charges for (a config passed through unresolved prices
	// no KV at all and would be accepted here).
	pinProbeFreeMemoryGB(t, gibFromMiB(2311))
	if err := modelLoadFitError(m, resolved); err == nil {
		t.Error("the guard must price the context the resolver fills, not zero KV")
	}
	// And the args say the same thing out loud instead of leaving the window to
	// the server.
	args, err := modelDirectArgs("legacy-model", m, resolved)
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := argValue(args, "--ctx-size"); !ok || v != "4096" {
		t.Errorf("--ctx-size = %q (present %v), want the filled default: %v", v, ok, args)
	}
}

// pinDirectModelDir points the model scan at an empty temp dir, clears the
// imported directory and the stored per-model configs, so scanModels sees
// exactly what the test wrote and every model counts as untouched.
func pinDirectModelDir(t *testing.T) string {
	t.Helper()
	saveConfigState(t)
	dir := t.TempDir()
	modelDownloadDirMu.Lock()
	modelDownloadDirOverride = dir
	modelDownloadDirMu.Unlock()
	modelsDirMu.Lock()
	customModelsDir = ""
	modelsDirMu.Unlock()
	modelConfigsMu.Lock()
	cachedModelConfigs = make(map[string]ModelConfig)
	modelConfigsMu.Unlock()
	return dir
}

// TestAndroidDirectStartRefusesOversizedModel covers the guard's placement on
// the Android direct-mode start path: a load bigger than the available memory
// is refused with the measured numbers before llama-server is spawned, so the
// device's low-memory killer is never reached (the historical symptom was the
// app process dying without an error).
func TestAndroidDirectStartRefusesOversizedModel(t *testing.T) {
	withPlatformGOOS(t, "android")
	// 2 KiB of weights against a ~1.5 KiB probe: the ratio, not the scale, is
	// what the branch under test keys on.
	dir := pinDirectModelDir(t)
	makeVariant(t, dir, "author", "huge-model", "huge.gguf", 2048)
	pinProbeFreeMemoryGB(t, 1.5e-6)

	err := startServerInternalWithModel("huge-model")
	if err == nil {
		t.Fatal("start must refuse a model that cannot fit in available memory")
	}
	// The figure in the message is the whole planned footprint, not the weight
	// file: 2 KiB + the fallback KV at the app's default 4096-token context +
	// the runtime overhead at the plan's ubatch.
	want := int64(2048) + 32*1024*4096 + int64(tuneRuntimeOverheadMB)*(1<<20)
	if !strings.Contains(err.Error(), formatBytes(want)) {
		t.Errorf("error %q must carry the planned load size %s", err, formatBytes(want))
	}
	serverMu.Lock()
	running, cmd := serverRunning, serverCmd
	serverMu.Unlock()
	if running || cmd != nil {
		t.Errorf("refused start must not spawn a server: running=%v cmd=%v", running, cmd)
	}
}

// TestAndroidDirectStartNoopSkipsFitCheck verifies a repeat start of the
// already-resident model stays a no-op even when the memory probe says it would
// not fit today: refusing it would only turn a working server into an error.
func TestAndroidDirectStartNoopSkipsFitCheck(t *testing.T) {
	withPlatformGOOS(t, "android")
	dir := pinDirectModelDir(t)
	makeVariant(t, dir, "author", "huge-model", "huge.gguf", 2048)
	pinProbeFreeMemoryGB(t, 1.5e-6)

	serverMu.Lock()
	serverRunning = true
	directServingModel = "huge-model"
	serverMu.Unlock()
	t.Cleanup(func() {
		serverMu.Lock()
		serverRunning = false
		directServingModel = ""
		serverMu.Unlock()
	})

	if err := startServerInternalWithModel("huge-model"); err != nil {
		t.Errorf("no-op start of the resident model must stay a no-op, got %v", err)
	}
	serverMu.Lock()
	running := serverRunning
	serverMu.Unlock()
	if !running {
		t.Error("no-op start must not stop the running server")
	}
}

// geomGGUF writes a readable GGUF header with a small exact KV geometry: 4
// layers, 512 bytes of f16 KV per token per layer (4 layers x 512 B = 2 MiB per
// 4096 tokens), trained context 32768. Every byte figure the tests below assert
// is arithmetic from these keys.
func geomGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	return writeTempGGUF(t, dir, name, buildGGUF(3,
		strKV("general.architecture", "qwen2"),
		u32KV("qwen2.block_count", 4),
		u64KV("qwen2.context_length", 32768),
		u32KV("qwen2.embedding_length", 512),
		u32KV("qwen2.attention.head_count", 4),
		u32KV("qwen2.attention.head_count_kv", 1),
		u32KV("qwen2.attention.key_length", 128),
		u32KV("qwen2.attention.value_length", 128),
	))
}

// mib is the MiB multiplier the load estimates are asserted in.
func mib(n int64) int64 { return n * (1 << 20) }

// TestPlanResidentSet prices what router mode may hold in memory at the same
// time: the modelsMax heaviest entries of the library, summed. Models load
// lazily, so the launch-time resident set proves nothing, and the cap is a count
// rather than a memory bound — measured with b11720 on a 24 GiB M2, two models
// really are resident at once (RSS 3,211 -> 3,663 MiB across worker processes
// before the third request evicted the biggest). The case that matters for the
// untuned model is the one the preset writer resolves: the writer emits
// routerUntunedCtx's number, so pricing it at the settings page's 4096 default
// would under-charge it by 56 MiB on this fixture. The probe is pinned so the
// expectation does not follow the test host's free RAM.
func TestPlanResidentSet(t *testing.T) {
	pinCPUSHapedReserve(t)
	pinProbeFreeMemoryGB(t, 64)
	dir := t.TempDir()
	path := geomGGUF(t, dir, "geom.gguf")
	weights := int64(1 << 30) // 1024 MiB
	overhead := mib(1280)     // runtimeOverheadBytes at the default ubatch

	if got := planResidentSet(nil, nil, 4); got.total != 0 || len(got.loads) != 0 || got.heaviest.model.Name != "" || got.heaviest.cfg.CtxSize != 0 {
		t.Errorf("empty library = %+v, want a zero set", got)
	}

	untuned := ModelInfo{Name: "untuned", Path: path, SizeBytes: weights}
	tuned := ModelInfo{Name: "tuned", Path: path, SizeBytes: weights}
	cfgs := map[string]ModelConfig{"tuned": {CtxSize: 4096, UBatchSize: 512}}

	// 1024 weights + 8 KV (4096 tokens) + 1280 overhead.
	if got := modelLoadBytes(tuned, cfgs["tuned"]); got != weights+mib(8)+overhead {
		t.Errorf("tuned ctx 4096 = %d MiB, want %d", got/(1<<20), (weights+mib(8)+overhead)/(1<<20))
	}
	// The untuned entry must be priced at the trained context (64 MiB of KV),
	// not at the app's 4096 default.
	if got := planResidentSet([]ModelInfo{untuned}, nil, 1).total; got != weights+mib(64)+overhead {
		t.Errorf("untuned model = %d MiB, want %d (trained ctx 32768)", got/(1<<20), (weights+mib(64)+overhead)/(1<<20))
	}

	// Three entries, three prices: the stored 131072-token config is heaviest
	// (1024 weights + 256 KV + 1280 overhead = 2560 MiB), then the
	// trained-context fallback (2368), then the stored 4096 plan (2312).
	big := ModelInfo{Name: "big", Path: path, SizeBytes: weights}
	cfgs["big"] = ModelConfig{CtxSize: 131072, UBatchSize: 512}
	library := []ModelInfo{tuned, untuned, big}
	const (
		bigPrice     = int64(2560)
		untunedPrice = int64(2368)
		tunedPrice   = int64(2312)
	)

	// At the app's default cap of one model the set is exactly the single
	// heaviest load, the figure these decisions used before the cap existed.
	one := planResidentSet(library, cfgs, 1)
	if one.total != mib(bigPrice) || len(one.loads) != 1 {
		t.Errorf("cap 1 total = %d MiB over %d loads, want %d over 1", one.total/(1<<20), len(one.loads), bigPrice)
	}
	// The same call names the model it belongs to and the plan it was priced
	// with, which is what the over-commit warning has to quote and re-price a
	// fitting tier against.
	if one.heaviest.model.Name != "big" {
		t.Errorf("heaviest model = %q, want the stored 131072-token entry", one.heaviest.model.Name)
	}
	if one.heaviest.cfg.CtxSize != 131072 || one.heaviest.cfg.UBatchSize != 512 {
		t.Errorf("heaviest plan = %+v, want the stored ctx 131072 at ubatch 512", one.heaviest.cfg)
	}

	// Raising the cap raises the bill by the next-heaviest load, heaviest first.
	two := planResidentSet(library, cfgs, 2)
	if two.total != mib(bigPrice+untunedPrice) {
		t.Errorf("cap 2 total = %d MiB, want %d (the two heaviest summed)", two.total/(1<<20), bigPrice+untunedPrice)
	}
	if len(two.loads) != 2 || two.loads[0].model.Name != "big" || two.loads[1].model.Name != "untuned" {
		t.Errorf("cap 2 loads = %+v, want big then untuned", two.loads)
	}
	all := planResidentSet(library, cfgs, 3)
	if all.total != mib(bigPrice+untunedPrice+tunedPrice) {
		t.Errorf("cap 3 total = %d MiB, want %d", all.total/(1<<20), bigPrice+untunedPrice+tunedPrice)
	}
	// A cap wider than the library cannot bill models that are not there.
	wide := planResidentSet(library, cfgs, 8)
	if wide.total != all.total || len(wide.loads) != 3 {
		t.Errorf("cap 8 = %d MiB over %d loads, want the same %d MiB over 3", wide.total/(1<<20), len(wide.loads), all.total/(1<<20))
	}
	// The clamp the app's own call site applies is the planner's too.
	if clamp := planResidentSet(library, cfgs, 0); clamp.total != one.total || len(clamp.loads) != 1 {
		t.Errorf("cap 0 = %+v, want the single heaviest load", clamp.loads)
	}
}

// gibFromMiB converts a MiB figure into the GiB value probeFreeMemoryGB returns,
// so memory thresholds can be written in the same unit the estimates are asserted
// in.
func gibFromMiB(n int64) float64 { return float64(n) / 1024 }

// TestRouterUntunedCtx pins the context an unconfigured model gets in router
// mode. geomGGUF trains at 32768 and costs 2 KiB of f16 KV per token, so with
// 1024 MiB of weights and the 1280 MiB runtime reserve its trained context prices
// at 2368 MiB and the ladder rungs below at 2336 (16384), 2320 (8192), 2312 (4096)
// and 2308 (2048).
func TestRouterUntunedCtx(t *testing.T) {
	pinCPUSHapedReserve(t)
	dir := t.TempDir()
	path := geomGGUF(t, dir, "geom.gguf")
	m := ModelInfo{Name: "untuned", Path: path, SizeBytes: 1 << 30}

	cases := []struct {
		name string
		gb   float64
		want int
	}{
		{"a machine that can pay keeps the trained context", gibFromMiB(4096), 32768},
		{"one MiB short of the trained price drops a rung", gibFromMiB(2340), 16384},
		{"cramped memory walks down to the rung that fits", gibFromMiB(2310), 2048},
		{"nothing affordable still gets the ladder floor", gibFromMiB(2300), 2048},
		{"a failed probe lands on the app default", 0, 4096},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pinProbeFreeMemoryGB(t, c.gb)
			if got := routerUntunedCtx(m); got != c.want {
				t.Errorf("routerUntunedCtx with %d MiB available = %d, want %d", int64(c.gb*1024), got, c.want)
			}
		})
	}

	// A model trained past the tuner's ceiling still stops at it: an untuned
	// entry must not carry a bigger window than the tuned plan for the same
	// model would. writeMemfitGGUF trains at 262,144 and costs 48 KiB of f16 KV
	// per token, so the ceiling prices at 8,448 MiB (1,024 + 6,144 + 1,280) and
	// the full metadata window at 14,592 MiB.
	pinProbeFreeMemoryGB(t, gibFromMiB(8700))
	wide := ModelInfo{Name: "wide", Path: writeMemfitGGUF(t, dir), SizeBytes: 1 << 30}
	if got := routerUntunedCtx(wide); got != tuneCtxMax {
		t.Errorf("262144-trained model with 8700 MiB available = %d, want the %d ceiling", got, tuneCtxMax)
	}

	// Unreadable metadata cannot be priced against a trained context, so the
	// app default is written instead of leaving the entry without one.
	pinProbeFreeMemoryGB(t, 64)
	blind := ModelInfo{Name: "blind", Path: filepath.Join(dir, "missing.gguf"), SizeBytes: 1 << 30}
	if got := routerUntunedCtx(blind); got != 4096 {
		t.Errorf("unreadable metrics = %d, want the 4096 app default", got)
	}
}

// TestUnconfiguredModelKeepsLibraryPromptCache is the regression for the harm the
// bounded context removes: with the untuned entry left without ctx-size, its
// trained context priced above the machine's memory made the library's heaviest
// load exceed available RAM, and the router-mode --cache-ram bound collapsed to 0
// for every model in the app. Priced at the largest context the machine affords,
// the same library keeps a cache budget.
func TestUnconfiguredModelKeepsLibraryPromptCache(t *testing.T) {
	pinCPUSHapedReserve(t)
	pinProbeFreeMemoryGB(t, gibFromMiB(2340))
	path := geomGGUF(t, t.TempDir(), "geom.gguf")
	m := ModelInfo{Name: "untuned", Path: path, SizeBytes: 1 << 30}

	heavy := planResidentSet([]ModelInfo{m}, nil, 1).total
	if want := mib(1024 + 32 + 1280); heavy != want {
		t.Errorf("heaviest = %d MiB, want %d (bounded ctx 16384)", heavy/(1<<20), want/(1<<20))
	}
	if got := promptCacheRAMMiB(8192, heavy, availableMemoryBytes()); got <= 0 {
		t.Errorf("prompt cache budget = %d MiB, want a positive bound over the affordable load", got)
	}
	// The figure this replaces: the same machine, priced at the trained context.
	if got := promptCacheRAMMiB(8192, mib(2368), availableMemoryBytes()); got != 0 {
		t.Errorf("trained-context price = %d MiB budget, want 0 (the collapse the bound removes)", got)
	}
}

// TestLoadsStayInHostRAM pins where modelLoadBytes' host-RAM figure is honest
// enough to warn from. It is on unified memory and on a CPU-only host; a
// discrete GPU that can actually hold the load takes weights and KV out of host
// memory, so the same figure over-charges there and a warning would be wrong.
func TestLoadsStayInHostRAM(t *testing.T) {
	tests := []struct {
		name string
		gpu  tuneTarget
		stay bool
	}{
		{"no GPU", tuneTarget{Vendor: vendorNone}, true},
		{"Apple Silicon with the unified pool listed", tuneTarget{Vendor: vendorApple, VRAMMB: 16384}, true},
		{"Apple Silicon reported without VRAM", tuneTarget{Vendor: vendorApple}, true},
		{"CUDA card too small to offload into", tuneTarget{Vendor: vendorNvidia, VRAMMB: tuneGPUActiveMinVRAMMB - 1}, true},
		{"CUDA card that holds the load", tuneTarget{Vendor: vendorNvidia, VRAMMB: 8192}, false},
		{"ROCm card that holds the load", tuneTarget{Vendor: vendorAMD, VRAMMB: 8192}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := loadsStayInHostRAM(tt.gpu); got != tt.stay {
				t.Errorf("loadsStayInHostRAM(%+v) = %v, want %v", tt.gpu, got, tt.stay)
			}
		})
	}
}

// TestOvercommitWarning walks the warning's boundary: it stays quiet for a
// library that fits (including exactly at the limit), for a failed probe and for
// an unreadable library, and otherwise quotes the model plus both figures - the
// state it describes otherwise leaves no diagnostic anywhere - plus the context
// tier the caller resolved, when one exists. With more than one load able to be
// resident it also says how many and quotes the heaviest one's own share, so the
// line names both the total the session can reach and the model to shrink.
func TestOvercommitWarning(t *testing.T) {
	const gib = int64(1 << 30)
	tests := []struct {
		name      string
		model     string
		resident  int
		need      int64
		heaviest  int64
		avail     int64
		suggest   int
		want      bool
		wantTier  bool
		wantCount bool
	}{
		{"fits", "m", 1, 3 * gib, 3 * gib, 5 * gib, 0, false, false, false},
		{"exactly at the limit fits", "m", 1, 4 * gib, 4 * gib, 4 * gib, 0, false, false, false},
		{"failed probe says nothing", "m", 2, 20 * gib, 12 * gib, 0, 16384, false, false, false},
		{"empty library says nothing", "", 1, 20 * gib, 20 * gib, gib, 16384, false, false, false},
		{"over-committed warns", "qwen-4b", 1, 4*gib + 1, 4*gib + 1, 4 * gib, 0, true, false, false},
		{"over-committed warns with the tier", "qwen-4b", 1, 4*gib + 1, 4*gib + 1, 4 * gib, 32768, true, true, false},
		{"the resident set over-commits", "qwen-4b", 3, 11 * gib, 4 * gib, 8 * gib, 0, true, false, true},
		{"a resident set with a tier that fits", "qwen-4b", 2, 9 * gib, 5 * gib, 8 * gib, 16384, true, true, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := overcommitWarning(tt.model, tt.resident, tt.need, tt.heaviest, tt.avail, tt.suggest)
			if tt.want && got == "" {
				t.Error("overcommitWarning = \"\", want a warning")
			}
			if !tt.want && got != "" {
				t.Errorf("overcommitWarning = %q, want none", got)
			}
			if !tt.want {
				return
			}
			frags := []string{tt.model, formatBytes(tt.need), formatBytes(tt.avail)}
			if tt.resident > 1 {
				// The heaviest load's own figure is the one the advice is about;
				// printing only the total would hide which model to open.
				frags = append(frags, formatBytes(tt.heaviest))
			}
			for _, frag := range frags {
				if !strings.Contains(got, frag) {
					t.Errorf("warning %q does not carry %q", got, frag)
				}
			}
			// Asserted by digits: the advice sentence is tr()-resolved.
			if tt.wantTier && !strings.Contains(got, strconv.Itoa(tt.suggest)) {
				t.Errorf("warning %q does not quote the fitting tier %d", got, tt.suggest)
			}
			if !tt.wantTier && tt.suggest > 0 && strings.Contains(got, strconv.Itoa(tt.suggest)) {
				t.Errorf("warning %q quotes a tier the caller did not resolve", got)
			}
			if tt.wantCount && !strings.Contains(got, strconv.Itoa(tt.resident)) {
				t.Errorf("warning %q does not say how many models can be resident", got)
			}
		})
	}
}

// TestFittingCtxRung pins the advice half of the memory check: the largest tier
// of the tuner's own ladder that this model would fit with its own settings -
// and never a tier at or above the context it already has.
//
// The writeMemfitGGUF fixture trains at 262,144 tokens and costs 48 KiB of f16 KV
// per token, so a 1 GiB model at the default ubatch prices at (weights + ctx x 48
// KiB + the 1280 MiB runtime reserve):
//
//	131072 → 8448 MiB   65536 → 5376 MiB   32768 → 3840 MiB   16384 → 3072 MiB
//	 8192  → 2688 MiB    4096 → 2496 MiB    2048 → 2400 MiB
//
// and the same fixture with a pinned q8_0 K+V pair at 17/32 of each KV term
// (131072 → 5568, 65536 → 3936, 32768 → 3120, 16384 → 2712, 8192 → 2508,
// 4096 → 2406, 2048 → 2355).
func TestFittingCtxRung(t *testing.T) {
	pinCPUSHapedReserve(t)
	dir := t.TempDir()
	readable := writeMemfitGGUF(t, dir)
	unreadable := writeTempGGUF(t, dir, "junk.gguf", make([]byte, 64))
	const weights = int64(1 << 30) // 1024 MiB
	m := ModelInfo{Name: "memfit", Path: readable, SizeBytes: weights}
	blind := ModelInfo{Name: "blind", Path: unreadable, SizeBytes: weights}
	const fullCtx = tuneCtxMax // 131072, the ladder's top rung

	tests := []struct {
		name  string
		model ModelInfo
		cfg   ModelConfig
		avail int64
		want  int
	}{
		{
			name:  "the tier that fits is named",
			model: m,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512},
			avail: mib(5376),
			want:  65536,
		},
		{
			// The suggestion is priced by the same modelLoadBytes as the
			// refusal, so it must respect that boundary too.
			name:  "one MiB short of a tier walks down another rung",
			model: m,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512},
			avail: mib(5375),
			want:  32768,
		},
		{
			// The refused load keeps its cache types: 65536 costs 5376 MiB of
			// f16 KV but 3936 MiB with the pair the guard prices at 17/32, so
			// the same memory affords a tier higher than the f16 ladder would.
			name:  "the pinned q8_0 pair is priced into the tier",
			model: m,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512, CacheTypeK: "q8_0", CacheTypeV: "q8_0"},
			avail: mib(4000),
			want:  65536,
		},
		{
			// Ample memory changes nothing about the direction: the helper only
			// ever suggests below the configured context, because the caller
			// reaches it with a load that price already refused.
			name:  "a tier above the configured context is never suggested",
			model: m,
			cfg:   ModelConfig{CtxSize: 16384, UBatchSize: 512},
			avail: mib(64 * 1024),
			want:  8192,
		},
		{
			// A hand-typed context sits between rungs; the largest tier below
			// it is still a reduction.
			name:  "an off-ladder context suggests the tier under it",
			model: m,
			cfg:   ModelConfig{CtxSize: 20000, UBatchSize: 512},
			avail: mib(3072),
			want:  16384,
		},
		{
			name:  "a context at the ladder floor has nothing left to suggest",
			model: m,
			cfg:   ModelConfig{CtxSize: tuneCtxMin, UBatchSize: 512},
			avail: mib(8704),
			want:  0,
		},
		{
			// The floor prices at 2400 MiB for this fixture; a machine that
			// cannot pay it is not helped by any tier.
			name:  "less than the floor price leaves no tier",
			model: m,
			cfg:   ModelConfig{CtxSize: 4096, UBatchSize: 512},
			avail: mib(2399),
			want:  0,
		},
		{
			// Weights and the runtime reserve alone already exceed the memory:
			// the advice has to be "a smaller model", not a number.
			name:  "weights alone over memory leaves no tier",
			model: m,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512},
			avail: mib(2000),
			want:  0,
		},
		{
			name:  "unknown availability suggests nothing",
			model: m,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512},
			avail: 0,
			want:  0,
		},
		{
			// Unreadable metadata is priced with the tuner's fallback geometry
			// (32 KiB/token, so 131072 → 6400 MiB and 65536 → 4352 MiB) - the
			// same figure the refusal quoted, so the advice stays consistent
			// with the bill.
			name:  "unreadable metadata keeps the fallback geometry",
			model: blind,
			cfg:   ModelConfig{CtxSize: fullCtx, UBatchSize: 512},
			avail: mib(5000),
			want:  65536,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := fittingCtxRung(tt.model, tt.cfg, tt.avail); got != tt.want {
				t.Errorf("fittingCtxRung(ctx %d, cacheK %q, avail %d MiB) = %d, want %d",
					tt.cfg.CtxSize, tt.cfg.CacheTypeK, tt.avail/(1<<20), got, tt.want)
			}
		})
	}
}
