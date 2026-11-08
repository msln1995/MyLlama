package core

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// mediaJobsTestEnv swaps every media seam into a controlled temp layout and
// installs a fake sd-cli script of the given behavior. mode is one of:
// ok (write PNG, exit 0), fakeok (exit 0, no file), fail (exit 3),
// hang (sleep 30s).
func mediaJobsTestEnv(t *testing.T, mode string) (outDir, engineDir string) {
	t.Helper()
	root := t.TempDir()
	engineDir = filepath.Join(root, "engine")
	outDir = filepath.Join(root, "outputs")
	srcDir := filepath.Join(root, "src")
	for _, d := range []string{engineDir, outDir, srcDir} {
		if err := os.MkdirAll(d, 0755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}

	// Fake engine: handles --version for the probe and behaves per mode.
	script := "#!/bin/sh\n" +
		"if [ \"$1\" = \"--version\" ]; then echo 'stable-diffusion.cpp version unknown, commit deadbeef17'; exit 0; fi\n" +
		"echo \"fake-engine mode=" + mode + " args:$#\"\n" +
		"OUT=\"\"\n" +
		"prev=\"\"\n" +
		"for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && OUT=\"$a\"; prev=\"$a\"; done\n" +
		"case " + mode + " in\n" +
		"  ok) sleep 0.15; cp \"$PNG_FIXTURE\" \"$OUT\"; exit 0 ;;\n" +
		"  fakeok) sleep 0.15; exit 0 ;;\n" +
		"  fail) echo 'boom: invalid weights' >&2; exit 3 ;;\n" +
		"  hang) sleep 30; exit 0 ;;\n" +
		"esac\n" +
		"exit 9\n"
	bin := filepath.Join(engineDir, "sd-cli")
	if err := os.WriteFile(bin, []byte(script), 0755); err != nil {
		t.Fatalf("write engine script: %v", err)
	}

	// Valid tiny PNG fixture (via image/png so the bytes are genuinely valid).
	img := image.NewRGBA(image.Rect(0, 0, 8, 6))
	for x := 0; x < 8; x++ {
		for y := 0; y < 6; y++ {
			img.Set(x, y, color.RGBA{R: 200, G: 10, B: 30, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode fixture: %v", err)
	}
	fixture := filepath.Join(root, "fixture.png")
	if err := os.WriteFile(fixture, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	t.Setenv("PNG_FIXTURE", fixture)

	// Engine + outputs settings point into the temp layout.
	mediaSettingsMu.Lock()
	oldEngine := mediaEngineDirSetting
	oldOut := mediaOutputsSetting
	mediaEngineDirSetting = engineDir
	mediaOutputsSetting = outDir
	mediaSettingsMu.Unlock()

	// Probe cache reset (engine dir changed).
	mediaResetRuntimeCache()

	// Faster test timing.
	oldGrace := mediaKillGrace
	mediaKillGrace = 500 * time.Millisecond
	oldTimeout := mediaJobTimeout
	mediaJobTimeout = 30 * time.Second

	// Controlled variant: two tiny verified components.
	f1 := filepath.Join(srcDir, "diffusion.bin")
	if err := os.WriteFile(f1, []byte("DIFFUSION-WEIGHTS-CONTENT"), 0644); err != nil {
		t.Fatalf("write comp: %v", err)
	}
	f2 := filepath.Join(srcDir, "te.bin")
	if err := os.WriteFile(f2, []byte("TE-CONTENT"), 0644); err != nil {
		t.Fatalf("write comp: %v", err)
	}
	sum1, _ := fileSHA256(f1)
	sum2, _ := fileSHA256(f2)
	comps := []MediaComponent{
		{Role: MediaRoleDiffusion, Repo: "t/r", Revision: "rev", File: "diffusion.bin", Size: 25, SHA256: sum1},
		{Role: MediaRoleTextEncoder, Repo: "t/r2", Revision: "rev", File: "te.bin", Size: 10, SHA256: sum2},
	}
	oldLookup := mediaVariantLookup
	mediaVariantLookup = func(modelID, variantID string) (*MediaModel, *MediaVariant, error) {
		if modelID != "m" || variantID != "v" {
			return nil, nil, fmt.Errorf("unknown model %s/%s", modelID, variantID)
		}
		return &MediaModel{ID: "m"}, &MediaVariant{
			ID: "v", Label: "test", Components: comps,
			Profile: MediaRunnerProfile{Family: "z-image-turbo", Kind: "image", CFGScale: 1.0, Steps: 8, Sampler: "euler", MinSize: 64, MaxSize: 1024, SizeMultiple: 8},
		}, nil
	}

	// Installs/jobs/works files in temp; fresh manager; idle arbiter.
	oldInstalls := mediaInstallsPath
	mediaInstallsPath = func() string { return filepath.Join(root, "installs.json") }
	oldJobs := mediaJobsPath
	mediaJobsPath = func() string { return filepath.Join(root, "jobs.json") }
	oldWorks := mediaWorksPath
	mediaWorksPath = func() string { return filepath.Join(root, "works.json") }
	if _, err := MediaImportVariant("m", "v", map[string]string{
		MediaRoleDiffusion: f1, MediaRoleTextEncoder: f2,
	}); err != nil {
		t.Fatalf("seed install: %v", err)
	}
	oldMgr := mediaJobsMgr
	mediaJobsMgr = &mediaJobManager{}
	// Reset the arbiter to a fresh zero value (never copy the mutex by value).
	mediaArbiter = inferenceArbiter{}

	t.Cleanup(func() {
		// Terminate any engine the test left behind (bounded, graceful-first),
		// then let the job goroutine's tail (works write, arbiter release)
		// fully settle before restoring the globals it may still touch.
		mediaJobsShutdown()
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) && len(ArbiterMediaHolders()) > 0 {
			time.Sleep(10 * time.Millisecond)
		}
		time.Sleep(100 * time.Millisecond)
		mediaSettingsMu.Lock()
		mediaEngineDirSetting = oldEngine
		mediaOutputsSetting = oldOut
		mediaSettingsMu.Unlock()
		mediaKillGrace = oldGrace
		mediaJobTimeout = oldTimeout
		mediaVariantLookup = oldLookup
		mediaInstallsPath = oldInstalls
		mediaJobsPath = oldJobs
		mediaWorksPath = oldWorks
		mediaJobsMgr = oldMgr
		mediaArbiter = inferenceArbiter{}
		mediaResetRuntimeCache()
	})
	return outDir, engineDir
}

// waitTerminal polls the manager until the job reaches a terminal status.
func waitTerminal(t *testing.T, jobID string) *MediaJob {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		for _, j := range GetMediaJobs() {
			if j.ID == jobID && mediaJobTerminal(j.Status) {
				return &j
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("job %s did not reach a terminal state in time", jobID)
	return nil
}

func mediaTestSubmit(t *testing.T, requestID string) *MediaJob {
	t.Helper()
	j, err := StartMediaJob(MediaJobStartRequest{
		RequestID: requestID, ModelID: "m", VariantID: "v",
		Params: MediaJobParams{Prompt: "a test cat", Seed: 42, Width: 64, Height: 64, Steps: 8, CFGScale: 1.0},
	})
	if err != nil {
		t.Fatalf("StartMediaJob: %v", err)
	}
	return j
}

func TestMediaJobSuccessLifecycle(t *testing.T) {
	outDir, _ := mediaJobsTestEnv(t, "ok")
	j := mediaTestSubmit(t, "req-ok")
	done := waitTerminal(t, j.ID)
	if done.Status != MediaJobSucceeded {
		t.Fatalf("status = %s (err=%s), want succeeded; log=%v", done.Status, done.Error, done.LogTail)
	}
	if done.Artifact == nil {
		t.Fatalf("artifact missing on success")
	}
	if done.Artifact.Width != 8 || done.Artifact.Height != 6 {
		t.Fatalf("artifact dims = %dx%d, want 8x6", done.Artifact.Width, done.Artifact.Height)
	}
	if done.EngineCommit != "deadbeef17" {
		t.Fatalf("engine commit = %q, want deadbeef17", done.EngineCommit)
	}
	if len(done.LogTail) == 0 {
		t.Fatalf("log tail should capture engine lines")
	}
	// Artifact resolvable from the works registry, file intact.
	p, err := MediaArtifactPath(done.Artifact.ID)
	if err != nil || !strings.HasPrefix(p, outDir) {
		t.Fatalf("artifact path = %s/%v", p, err)
	}
	// Reservation released after terminal state.
	if ArbiterMediaBlocked() {
		t.Fatalf("media reservation should be released after completion")
	}
	if ArbiterTryReserveMedia("post-check") != nil {
		t.Fatalf("arbiter should accept a new reservation after completion")
	}
	ArbiterReleaseMedia("post-check")
}

func TestMediaJobFakeSuccessFails(t *testing.T) {
	mediaJobsTestEnv(t, "fakeok")
	j := mediaTestSubmit(t, "req-fake")
	done := waitTerminal(t, j.ID)
	if done.Status != MediaJobFailed {
		t.Fatalf("status = %s, want failed (zero exit without artifact is never a success)", done.Status)
	}
	if !strings.Contains(done.Error, "artifact validation failed") {
		t.Fatalf("error = %q, want artifact validation failure", done.Error)
	}
	if done.Artifact != nil {
		t.Fatalf("failed job must not register an artifact")
	}
	if len(ListMediaArtifacts()) != 0 {
		t.Fatalf("works registry must stay empty")
	}
}

func TestMediaJobEngineFailure(t *testing.T) {
	mediaJobsTestEnv(t, "fail")
	j := mediaTestSubmit(t, "req-fail")
	done := waitTerminal(t, j.ID)
	if done.Status != MediaJobFailed {
		t.Fatalf("status = %s, want failed", done.Status)
	}
	if !strings.Contains(done.Error, "exited with code 3") {
		t.Fatalf("error = %q, want nonzero exit report", done.Error)
	}
}

func TestMediaJobCancelRunning(t *testing.T) {
	mediaJobsTestEnv(t, "hang")
	j := mediaTestSubmit(t, "req-hang")
	// Wait until the job is actually running before cancelling.
	deadline := time.Now().Add(10 * time.Second)
	for {
		snap := GetMediaJobs()
		if len(snap) > 0 && snap[0].ID == j.ID && snap[0].Status == MediaJobRunning {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("job never reached running (status=%s)", snap[0].Status)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := CancelMediaJob(j.ID); err != nil {
		t.Fatalf("cancel: %v", err)
	}
	done := waitTerminal(t, j.ID)
	if done.Status != MediaJobCancelled {
		t.Fatalf("status = %s, want cancelled", done.Status)
	}
	if done.Artifact != nil {
		t.Fatalf("cancelled job must not have an artifact")
	}
}

func TestMediaJobCancelQueued(t *testing.T) {
	mediaJobsTestEnv(t, "hang")
	first := mediaTestSubmit(t, "req-hang")
	// Wait for the first to occupy the slot, then the second is queued.
	deadline := time.Now().Add(10 * time.Second)
	for {
		if s := GetMediaJobs(); len(s) > 0 && s[0].ID == first.ID && s[0].Status == MediaJobRunning {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("first job never reached running")
		}
		time.Sleep(20 * time.Millisecond)
	}
	second := mediaTestSubmit(t, "req-queued")
	if second.Status != MediaJobQueued {
		t.Fatalf("second job status = %s, want queued", second.Status)
	}
	if err := CancelMediaJob(second.ID); err != nil {
		t.Fatalf("cancel queued: %v", err)
	}
	// The queued job reaches cancelled promptly (before the first finishes).
	deadline = time.Now().Add(5 * time.Second)
	for {
		for _, j := range GetMediaJobs() {
			if j.ID == second.ID && j.Status == MediaJobCancelled {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("queued job did not cancel in time")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestMediaJobIdempotentSubmit(t *testing.T) {
	mediaJobsTestEnv(t, "ok")
	a := mediaTestSubmit(t, "same-request")
	b := mediaTestSubmit(t, "same-request")
	if a.ID != b.ID {
		t.Fatalf("idempotent resubmission returned different jobs: %s vs %s", a.ID, b.ID)
	}
	waitTerminal(t, a.ID)
	// Re-submitting after completion still returns the original job.
	c := mediaTestSubmit(t, "same-request")
	if c.ID != a.ID || c.Status != MediaJobSucceeded {
		t.Fatalf("post-completion resubmit = %s/%s, want original succeeded job", c.ID, c.Status)
	}
}

func TestMediaJobRestartRecovery(t *testing.T) {
	mediaJobsTestEnv(t, "ok")
	// Craft a persisted history with an in-flight job, then boot a fresh
	// manager over it (the production startup path).
	f := &mediaJobsFile{SchemaVersion: mediaJobsSchemaVersion, Jobs: []MediaJob{
		{ID: "job-inflight", ModelID: "m", VariantID: "v", Status: MediaJobRunning, CreatedAt: time.Now().Add(-time.Minute)},
		{ID: "job-done", ModelID: "m", VariantID: "v", Status: MediaJobSucceeded, CreatedAt: time.Now().Add(-2 * time.Minute)},
	}}
	if err := saveMediaJobsFile(f); err != nil {
		t.Fatalf("seed jobs file: %v", err)
	}
	mediaJobsMgr = &mediaJobManager{}
	jobs := loadMediaJobs()
	if len(jobs) != 2 {
		t.Fatalf("loaded %d jobs, want 2", len(jobs))
	}
	byID := map[string]string{}
	for _, j := range jobs {
		byID[j.ID] = j.Status
	}
	if byID["job-inflight"] != MediaJobInterrupted {
		t.Fatalf("in-flight job recovered as %s, want interrupted", byID["job-inflight"])
	}
	if byID["job-done"] != MediaJobSucceeded {
		t.Fatalf("terminal job must not be touched, got %s", byID["job-done"])
	}
}

func TestMediaJobQueueFull(t *testing.T) {
	mediaJobsTestEnv(t, "hang")
	// One running + queue limit queued: the next submit must reject.
	first := mediaTestSubmit(t, "req-0")
	defer CancelMediaJob(first.ID)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if s := GetMediaJobs(); len(s) > 0 && s[0].Status == MediaJobRunning {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("first job never reached running")
		}
		time.Sleep(20 * time.Millisecond)
	}
	for i := 1; i <= mediaJobQueueLimit; i++ {
		if _, err := StartMediaJob(MediaJobStartRequest{
			RequestID: fmt.Sprintf("req-%d", i), ModelID: "m", VariantID: "v",
			Params: MediaJobParams{Prompt: "p", Seed: 1, Width: 64, Height: 64, Steps: 1, CFGScale: 1.0},
		}); err != nil {
			t.Fatalf("queued submit %d: %v", i, err)
		}
	}
	if _, err := StartMediaJob(MediaJobStartRequest{
		RequestID: "req-over", ModelID: "m", VariantID: "v",
		Params: MediaJobParams{Prompt: "p", Seed: 1, Width: 64, Height: 64, Steps: 1, CFGScale: 1.0},
	}); err == nil || !strings.Contains(err.Error(), "queue is full") {
		t.Fatalf("want queue-full rejection, got %v", err)
	}
}

func TestMediaJobLLamaRunningBlocks(t *testing.T) {
	mediaJobsTestEnv(t, "ok")
	ArbiterClaimLLama()
	defer ArbiterAbandonLLama()
	if _, err := StartMediaJob(MediaJobStartRequest{
		RequestID: "req-blocked", ModelID: "m", VariantID: "v",
		Params: MediaJobParams{Prompt: "p", Seed: 1, Width: 64, Height: 64, Steps: 1, CFGScale: 1.0},
	}); err == nil || !IsLLamaBusy(err) {
		t.Fatalf("want llama-busy rejection, got %v", err)
	}
}

func TestValidateMediaJobParams(t *testing.T) {
	profile := MediaRunnerProfile{Kind: "image", MinSize: 256, MaxSize: 1024, SizeMultiple: 16, Steps: 8, Sampler: "euler"}
	cases := []struct {
		name    string
		p       MediaJobParams
		wantErr string
	}{
		{"ok", MediaJobParams{Prompt: "x", Width: 512, Height: 512, Steps: 8, CFGScale: 1.0}, ""},
		{"empty prompt", MediaJobParams{Prompt: "  ", Width: 512, Height: 512, Steps: 8}, "prompt is required"},
		{"prompt too long", MediaJobParams{Prompt: strings.Repeat("a", 5000), Width: 512, Height: 512, Steps: 8}, "exceeds 4000"},
		{"below min size", MediaJobParams{Prompt: "x", Width: 128, Height: 512, Steps: 8}, "outside supported range"},
		{"above max size", MediaJobParams{Prompt: "x", Width: 512, Height: 2048, Steps: 8}, "outside supported range"},
		{"misaligned", MediaJobParams{Prompt: "x", Width: 500, Height: 512, Steps: 8}, "multiple of 16"},
		{"steps zero", MediaJobParams{Prompt: "x", Width: 512, Height: 512, Steps: 0}, "steps 0 outside"},
		{"steps too many", MediaJobParams{Prompt: "x", Width: 512, Height: 512, Steps: 500}, "outside 1..150"},
		{"cfg negative", MediaJobParams{Prompt: "x", Width: 512, Height: 512, Steps: 8, CFGScale: -1}, "outside 0..30"},
	}
	for _, tc := range cases {
		err := validateMediaJobParams(tc.p, profile)
		if tc.wantErr == "" && err != nil {
			t.Fatalf("%s: unexpected error %v", tc.name, err)
		}
		if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
			t.Fatalf("%s: want error containing %q, got %v", tc.name, tc.wantErr, err)
		}
	}
}

func TestParsePNGSize(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.png")
	img := image.NewRGBA(image.Rect(0, 0, 320, 200))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if err := os.WriteFile(p, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	w, h, err := parsePNGSize(p)
	if err != nil || w != 320 || h != 200 {
		t.Fatalf("parsePNGSize = %dx%d/%v, want 320x200", w, h, err)
	}
	// Corrupt signature is rejected, not guessed.
	bad := filepath.Join(dir, "b.png")
	if err := os.WriteFile(bad, []byte("definitely not a png........"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, err := parsePNGSize(bad); err == nil || !strings.Contains(err.Error(), "not a valid PNG") {
		t.Fatalf("want PNG rejection, got %v", err)
	}
}

func TestBuildMediaImageArgs(t *testing.T) {
	variant := &MediaVariant{
		ID: "v", Label: "L",
		Components: []MediaComponent{
			{Role: MediaRoleDiffusion, Repo: "r", Revision: "x", File: "d.gguf"},
			{Role: MediaRoleTextEncoder, Repo: "r2", Revision: "x", File: "t.gguf"},
			{Role: MediaRoleVAE, Repo: "r3", Revision: "x", File: "ae.safetensors"},
		},
		Profile: MediaRunnerProfile{Family: "z-image-turbo", Kind: "image", CFGScale: 1, Steps: 8, Sampler: "euler"},
	}
	install := &MediaInstallRecord{ModelID: "m", VariantID: "v", Components: []MediaComponentSource{
		{Role: MediaRoleDiffusion, Path: "/w/d.gguf"},
		{Role: MediaRoleTextEncoder, Path: "/w/t.gguf"},
		{Role: MediaRoleVAE, Path: "/w/ae.safetensors"},
	}}
	params := MediaJobParams{Prompt: "cat", Seed: 7, Width: 512, Height: 256, Steps: 8, CFGScale: 1.0}
	args, err := buildMediaImageArgs(variant, install, params, "/out/output.png")
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"-M img_gen",
		"--diffusion-model /w/d.gguf",
		"--llm /w/t.gguf",
		"--vae /w/ae.safetensors",
		"-p cat",
		"-H 256", "-W 512",
		"--steps 8", "--cfg-scale 1", "--seed 7",
		"-o /out/output.png",
		"--sampling-method euler",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args missing %q: %s", want, joined)
		}
	}
	// Missing install role fails closed.
	bad := *install
	bad.Components = bad.Components[:2]
	if _, err := buildMediaImageArgs(variant, &bad, params, "/o.png"); err == nil || !strings.Contains(err.Error(), "misses component role") {
		t.Fatalf("want missing-role failure, got %v", err)
	}
	// Unknown family fails closed.
	unk := *variant
	unk.Profile.Family = "mystery"
	if _, err := buildMediaImageArgs(&unk, install, params, "/o.png"); err == nil || !strings.Contains(err.Error(), "cannot wire") {
		t.Fatalf("want unknown-family failure, got %v", err)
	}
}

// TestMediaArbiterConcurrency hammers the gate: N concurrent media
// reservations all coexist (holder-set semantics), each releases by its own
// holder, and the llama side stays blocked while any holder remains.
func TestMediaArbiterConcurrency(t *testing.T) {
	mediaArbiter = inferenceArbiter{}
	defer func() { mediaArbiter = inferenceArbiter{} }()
	const n = 32
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := ArbiterTryReserveMedia(fmt.Sprintf("h%d", i)); err != nil {
				t.Errorf("reserve h%d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	if got := len(ArbiterMediaHolders()); got != n {
		t.Fatalf("holders = %d, want %d", got, n)
	}
	// Any media holder blocks a llama claim.
	if ArbiterClaimLLama() {
		t.Fatalf("llama claim must fail while media holds reservations")
	}
	// Releasing one of many keeps the llama side blocked.
	ArbiterReleaseMedia("h0")
	if ArbiterClaimLLama() {
		t.Fatalf("llama claim must fail while other holders remain")
	}
	// Release everything: gate reopens both ways.
	ArbiterReleaseMedia("")
	if got := len(ArbiterMediaHolders()); got != 0 {
		t.Fatalf("holders after full release = %d, want 0", got)
	}
	if !ArbiterClaimLLama() {
		t.Fatalf("llama claim must succeed with no media holders")
	}
	if err := ArbiterTryReserveMedia("x"); !IsLLamaBusy(err) {
		t.Fatalf("want llama-busy while claimed, got %v", err)
	}
	ArbiterAbandonLLama()
	if err := ArbiterTryReserveMedia("x"); err != nil {
		t.Fatalf("gate must reopen after llama abandon: %v", err)
	}
	ArbiterReleaseMedia("x")
}
