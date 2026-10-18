package core

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pinProbeFreeMemoryGB(t *testing.T, gb float64) {
	t.Helper()
	orig := probeFreeMemoryGB
	probeFreeMemoryGB = func() float64 { return gb }
	t.Cleanup(func() { probeFreeMemoryGB = orig })
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
// cannot fit produces an error carrying both measured figures.
func TestMemoryFitError(t *testing.T) {
	gib := int64(1 << 30)
	tests := []struct {
		name    string
		need    int64
		avail   int64
		wantErr bool
	}{
		{"fits", 3 * gib, 5 * gib, false},
		{"exactly at the limit fits", 4 * gib, 4 * gib, false},
		{"one byte over is refused", 4*gib + 1, 4 * gib, true},
		{"unknown availability never blocks", 20 * gib, 0, false},
		{"failed probe (negative) never blocks", 20 * gib, -1, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := memoryFitError("qwen-4b", tt.need, tt.avail)
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
		overhd  = int64(tuneCPURuntimeOverheadMB) * mib
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
			name: "same-directory mmproj counts as weights",
			m:    ModelInfo{Name: "m", Path: mmprojModel, SizeBytes: fourGib, HasMMProj: true},
			cfg:  ModelConfig{},
			want: fourGib + mmprojSize + overhd,
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

// TestModelLoadFitError verifies the scanned-model entry point against the
// probe: the overhead and the context are part of the decision, and a failed
// probe never blocks a start.
func TestModelLoadFitError(t *testing.T) {
	const (
		gib        = int64(1 << 30)
		overheadGB = float64(tuneCPURuntimeOverheadMB) / 1024.0 // 1.25 GiB
		fallbackGB = float64(32*1024*4096) / float64(1<<30)     // 0.125 GiB at ctx 4096
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
}

// TestDirectModelConfigFallsBackToAppDefaults pins the direct-mode config
// resolution: a stored config wins as-is, and a model the user never touched
// gets the app default rather than a zero value (which would leave
// --ctx-size unset and hand llama-server the model's trained context).
func TestDirectModelConfigFallsBackToAppDefaults(t *testing.T) {
	stored := ModelConfig{CtxSize: 32768, Threads: 4}
	cfgs := map[string]ModelConfig{"tuned": stored}

	if got := directModelConfig(cfgs, "tuned"); got != stored {
		t.Errorf("stored config = %+v, want %+v", got, stored)
	}
	if got := directModelConfig(cfgs, "untouched"); got.CtxSize != defaultModelConfig().CtxSize {
		t.Errorf("untouched model CtxSize = %d, want the app default %d", got.CtxSize, defaultModelConfig().CtxSize)
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
	// the CPU runtime overhead.
	want := int64(2048) + 32*1024*4096 + int64(tuneCPURuntimeOverheadMB)*(1<<20)
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
