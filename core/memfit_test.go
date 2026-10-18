package core

import (
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

// TestMemoryFitError walks the fit boundary: a model that fits (and one
// exactly at the limit) is accepted, a probe failure never blocks, and a model
// that cannot fit produces an error carrying both measured figures.
func TestMemoryFitError(t *testing.T) {
	gib := int64(1 << 30)
	tests := []struct {
		name    string
		weights int64
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
			err := memoryFitError("qwen-4b", tt.weights, tt.avail)
			if (err != nil) != tt.wantErr {
				t.Fatalf("memoryFitError(%d, %d) err = %v, wantErr %v", tt.weights, tt.avail, err, tt.wantErr)
			}
			if err == nil {
				return
			}
			// The message is the only thing the user gets: it must name the
			// model and both numbers, whatever UI language tr() resolved to.
			msg := err.Error()
			for _, want := range []string{"qwen-4b", formatBytes(tt.weights), formatBytes(tt.avail)} {
				if !strings.Contains(msg, want) {
					t.Errorf("error %q missing %q", msg, want)
				}
			}
		})
	}
}

// TestModelLoadFitError verifies the scanned-model entry point against the
// live probe: an ample machine accepts it, a cramped one refuses it.
func TestModelLoadFitError(t *testing.T) {
	gib := int64(1 << 30)
	m := ModelInfo{Name: "huge-model", SizeBytes: 8 * gib}

	pinProbeFreeMemoryGB(t, 10)
	if err := modelLoadFitError(m); err != nil {
		t.Errorf("10 GiB available must accept an 8 GiB model, got %v", err)
	}

	pinProbeFreeMemoryGB(t, 3)
	if err := modelLoadFitError(m); err == nil {
		t.Error("3 GiB available must refuse an 8 GiB model")
	}

	pinProbeFreeMemoryGB(t, 0)
	if err := modelLoadFitError(m); err != nil {
		t.Errorf("failed probe must not block a start, got %v", err)
	}
}

// pinDirectModelDir points the model scan at an empty temp dir and clears the
// imported directory, so scanModels sees exactly what the test wrote.
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
	return dir
}

// TestAndroidDirectStartRefusesOversizedModel covers the guard's placement on
// the Android direct-mode start path: a model bigger than the available memory
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
	if !strings.Contains(err.Error(), formatBytes(2048)) {
		t.Errorf("error %q must carry the model size", err)
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
