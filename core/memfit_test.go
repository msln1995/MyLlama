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

// TestModelLoadBytes pins the three terms of the planned footprint and how the
// KV term reacts to the model metadata and the configured context.
func TestModelLoadBytes(t *testing.T) {
	const (
		mib     = int64(1 << 20)
		gib     = int64(1 << 30)
		overhd  = int64(tuneRuntimeOverheadMB) * mib // priced at the default ubatch
		fourGib = 4 * gib
		// Unreadable metadata falls back to the tuner's conservative geometry:
		// 32 layers x 1024 B/token = 32 KiB per token.
		fallbackKV4096 = 32 * 1024 * 4096
		hybridKV4096   = 49152 * 4096
	)
	dir := t.TempDir()
	readable := writeMemfitGGUF(t, dir)
	unreadable := writeTempGGUF(t, dir, "junk.gguf", make([]byte, 64))
	mmprojDir := t.TempDir()
	mmprojModel := writeTempGGUF(t, mmprojDir, "vision.gguf", make([]byte, 64))
	const mmprojSize = int64(4096)
	if err := os.WriteFile(filepath.Join(mmprojDir, "mmproj-vision.gguf"), make([]byte, mmprojSize), 0o644); err != nil {
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := modelLoadBytes(tt.m, tt.cfg); got != tt.want {
				t.Errorf("modelLoadBytes = %d, want %d", got, tt.want)
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
	dir := t.TempDir()
	path := writeMemfitGGUF(t, dir)
	const fourGib = int64(4) * (1 << 30)
	m := ModelInfo{Name: "phone-model", Path: path, SizeBytes: fourGib}

	metrics, ok := readGGUFModelMetrics(path)
	if !ok {
		t.Fatal("fixture GGUF is not readable")
	}
	tm := buildTuneModel(metrics, ok, tuneWeightsBytes(m))
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
	dir := t.TempDir()
	path := writeMemfitGGUF(t, dir)
	m := ModelInfo{Name: "phone-model", Path: path, SizeBytes: 4 * (1 << 30)}

	metrics, ok := readGGUFModelMetrics(path)
	if !ok {
		t.Fatal("fixture GGUF is not readable")
	}
	tm := buildTuneModel(metrics, ok, tuneWeightsBytes(m))
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

// TestModelLoadFitError verifies the scanned-model entry point against the
// probe: the overhead and the context are part of the decision, and a failed
// probe never blocks a start.
func TestModelLoadFitError(t *testing.T) {
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

// TestMaxModelLoadBytes prices every entry of a library and keeps the heaviest,
// which is what the router-mode prompt-cache budget is bounded against (models
// load lazily, so the launch-time resident set proves nothing). The case that
// matters is the untuned model: the writer resolves its context through
// routerUntunedCtx, and on a machine that can pay for the trained 32768-token
// context pricing it at the settings page's 4096 default would under-charge it by
// 56 MiB on this fixture. The probe is pinned so the expectation does not follow
// the test host's free RAM.
func TestMaxModelLoadBytes(t *testing.T) {
	pinProbeFreeMemoryGB(t, 64)
	dir := t.TempDir()
	path := geomGGUF(t, dir, "geom.gguf")
	weights := int64(1 << 30) // 1024 MiB
	overhead := mib(1280)     // runtimeOverheadBytes at the default ubatch

	if got := maxModelLoadBytes(nil, nil); got != 0 {
		t.Errorf("empty library = %d, want 0", got)
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
	if got := maxModelLoadBytes([]ModelInfo{untuned}, nil); got != weights+mib(64)+overhead {
		t.Errorf("untuned model = %d MiB, want %d (trained ctx 32768)", got/(1<<20), (weights+mib(64)+overhead)/(1<<20))
	}
	// Heaviest of the three shapes wins, and a stored config can be heavier than
	// the trained-context fallback. The same call names the model it belongs to
	// and the plan it was priced with, which is what the over-commit warning has
	// to quote and re-price a fitting tier against.
	big := ModelInfo{Name: "big", Path: path, SizeBytes: weights}
	cfgs["big"] = ModelConfig{CtxSize: 131072, UBatchSize: 512}
	heavy, heavyPlan, got := heaviestPlannedLoad([]ModelInfo{tuned, untuned, big}, cfgs)
	if got != weights+mib(256)+overhead {
		t.Errorf("heaviest = %d MiB, want %d (stored ctx 131072)", got/(1<<20), (weights+mib(256)+overhead)/(1<<20))
	}
	if heavy.Name != "big" {
		t.Errorf("heaviest model = %q, want the stored 131072-token entry", heavy.Name)
	}
	if heavyPlan.CtxSize != 131072 || heavyPlan.UBatchSize != 512 {
		t.Errorf("heaviest plan = %+v, want the stored ctx 131072 at ubatch 512", heavyPlan)
	}
	if h, p, got := heaviestPlannedLoad(nil, nil); got != 0 || h.Name != "" || p.CtxSize != 0 {
		t.Errorf("empty library = %q/%+v/%d, want no model, no plan and 0", h.Name, p, got)
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
	pinProbeFreeMemoryGB(t, gibFromMiB(2340))
	path := geomGGUF(t, t.TempDir(), "geom.gguf")
	m := ModelInfo{Name: "untuned", Path: path, SizeBytes: 1 << 30}

	heavy := maxModelLoadBytes([]ModelInfo{m}, nil)
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
// tier the caller resolved, when one exists.
func TestOvercommitWarning(t *testing.T) {
	const gib = int64(1 << 30)
	tests := []struct {
		name     string
		model    string
		need     int64
		avail    int64
		suggest  int
		want     bool
		wantTier bool
	}{
		{"fits", "m", 3 * gib, 5 * gib, 0, false, false},
		{"exactly at the limit fits", "m", 4 * gib, 4 * gib, 0, false, false},
		{"failed probe says nothing", "m", 20 * gib, 0, 16384, false, false},
		{"empty library says nothing", "", 20 * gib, gib, 16384, false, false},
		{"over-committed warns", "qwen-4b", 4*gib + 1, 4 * gib, 0, true, false},
		{"over-committed warns with the tier", "qwen-4b", 4*gib + 1, 4 * gib, 32768, true, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := overcommitWarning(tt.model, tt.need, tt.avail, tt.suggest)
			if tt.want && got == "" {
				t.Error("overcommitWarning = \"\", want a warning")
			}
			if !tt.want && got != "" {
				t.Errorf("overcommitWarning = %q, want none", got)
			}
			if !tt.want {
				return
			}
			for _, frag := range []string{"qwen-4b", formatBytes(tt.need), formatBytes(tt.avail)} {
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
