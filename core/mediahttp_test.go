package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// mediaHTTPTestSetup reuses the jobs test env (fake engine of the given mode)
// and returns a handler to drive with httptest.
func mediaHTTPTestSetup(t *testing.T, mode string) (http.Handler, string) {
	t.Helper()
	outDir, _ := mediaJobsTestEnv(t, mode)
	return mediaHTTPMux(), outDir
}

// pngFixtureBytes returns a genuinely valid PNG (8x6) for artifact serving.
func pngFixtureBytes(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 6))
	img.Set(0, 0, color.RGBA{R: 1, G: 2, B: 3, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return buf.Bytes()
}

func TestMediaHTTPCapabilitiesImageOnly(t *testing.T) {
	h, _ := mediaHTTPTestSetup(t, "ok")
	req := httptest.NewRequest("GET", "/v1/media/capabilities", nil)
	req.Host = "127.0.0.1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("capabilities status = %d: %s", rec.Code, rec.Body.String())
	}
	var caps mediaCapabilities
	if err := json.Unmarshal(rec.Body.Bytes(), &caps); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(caps.Modes) != 1 || caps.Modes[0] != "image" {
		t.Fatalf("modes = %v, want [image] only (video is not open in P1)", caps.Modes)
	}
}

func TestMediaHTTPHostHeaderEnforced(t *testing.T) {
	h, _ := mediaHTTPTestSetup(t, "ok")
	req := httptest.NewRequest("GET", "/v1/media/capabilities", nil)
	req.Host = "evil.example.com"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("non-loopback Host status = %d, want 403", rec.Code)
	}
	// Loopback forms pass.
	for _, host := range []string{"127.0.0.1:1901", "localhost:1901", "127.0.0.1"} {
		req := httptest.NewRequest("GET", "/v1/media/capabilities", nil)
		req.Host = host
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("host %q status = %d, want 200", host, rec.Code)
		}
	}
}

func TestMediaHTTPSubmitPollArtifactFlow(t *testing.T) {
	h, _ := mediaHTTPTestSetup(t, "ok")
	// Submit.
	body := `{"requestId":"http-flow-1","modelId":"m","variantId":"v","params":{"prompt":"a cat","seed":42,"width":64,"height":64,"steps":8,"cfgScale":1.0}}`
	req := httptest.NewRequest("POST", "/v1/media/jobs", strings.NewReader(body))
	req.Host = "127.0.0.1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("submit status = %d: %s", rec.Code, rec.Body.String())
	}
	var job MediaJob
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatalf("decode job: %v", err)
	}
	// Poll to terminal.
	deadline := time.Now().Add(15 * time.Second)
	for {
		g := httptest.NewRecorder()
		greq := httptest.NewRequest("GET", "/v1/media/jobs/"+job.ID, nil)
		greq.Host = "127.0.0.1"
		h.ServeHTTP(g, greq)
		if g.Code != http.StatusOK {
			t.Fatalf("get job status = %d: %s", g.Code, g.Body.String())
		}
		var snap MediaJob
		_ = json.Unmarshal(g.Body.Bytes(), &snap)
		if mediaJobTerminal(snap.Status) {
			if snap.Status != MediaJobSucceeded {
				t.Fatalf("job status = %s (err=%s), want succeeded", snap.Status, snap.Error)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("job did not finish in time")
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Works list contains the artifact; file endpoint serves real PNG bytes.
	w := httptest.NewRecorder()
	wreq := httptest.NewRequest("GET", "/v1/media/works", nil)
	wreq.Host = "127.0.0.1"
	h.ServeHTTP(w, wreq)
	var works []MediaArtifact
	if err := json.Unmarshal(w.Body.Bytes(), &works); err != nil {
		t.Fatalf("decode works: %v", err)
	}
	if len(works) != 1 {
		t.Fatalf("works = %d, want 1", len(works))
	}
	f := httptest.NewRecorder()
	freq := httptest.NewRequest("GET", "/v1/media/works/"+works[0].ID+"/file", nil)
	freq.Host = "127.0.0.1"
	h.ServeHTTP(f, freq)
	if f.Code != http.StatusOK {
		t.Fatalf("artifact file status = %d", f.Code)
	}
	if ct := f.Header().Get("Content-Type"); ct != "image/png" {
		t.Fatalf("content type = %q, want image/png", ct)
	}
	if _, err := png.Decode(bytes.NewReader(f.Body.Bytes())); err != nil {
		t.Fatalf("served bytes are not a decodable PNG: %v", err)
	}
}

func TestMediaHTTPUnknownJob404(t *testing.T) {
	h, _ := mediaHTTPTestSetup(t, "ok")
	req := httptest.NewRequest("GET", "/v1/media/jobs/job-nope", nil)
	req.Host = "127.0.0.1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
}

func TestMediaHTTPBindConflictVisible(t *testing.T) {
	// A bind failure (port conflict) must surface as a loud "failed" state —
	// never a fake "running" (plan §5.7: 端口冲突必须可见且不能误报服务已启动).
	oldBind := mediaHTTPBind
	mediaHTTPBind = func(addr string) (net.Listener, error) {
		return nil, fmt.Errorf("address already in use (simulated conflict)")
	}
	mediaHTTPMu.Lock()
	mediaHTTPServer = nil
	mediaHTTPState = ""
	mediaHTTPMu.Unlock()
	defer func() {
		mediaHTTPStop()
		mediaHTTPBind = oldBind
	}()
	mediaHTTPStart()
	st := mediaHTTPStatusSnapshot()
	if st.Running || st.State != "failed" {
		t.Fatalf("state = %s running=%v, want failed and not running", st.State, st.Running)
	}
}

func TestMediaHTTPVideoVariantRejected(t *testing.T) {
	h, _ := mediaHTTPTestSetup(t, "ok")
	// Swap the variant lookup to a video profile: submission must fail with an
	// explicit "not open" error, not a silent success.
	oldLookup := mediaVariantLookup
	mediaVariantLookup = func(modelID, variantID string) (*MediaModel, *MediaVariant, error) {
		return &MediaModel{ID: "m"}, &MediaVariant{
			ID: "v", Profile: MediaRunnerProfile{Family: "wan", Kind: "video"},
		}, nil
	}
	defer func() { mediaVariantLookup = oldLookup }()
	body := `{"requestId":"http-video-1","modelId":"m","variantId":"v","params":{"prompt":"cat","seed":1,"width":64,"height":64,"steps":1,"cfgScale":1}}`
	req := httptest.NewRequest("POST", "/v1/media/jobs", strings.NewReader(body))
	req.Host = "127.0.0.1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("video submit status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "not open") {
		t.Fatalf("error = %s, want explicit not-open rejection", rec.Body.String())
	}
}

func TestMediaLoopbackHostOK(t *testing.T) {
	cases := []struct {
		host string
		ok   bool
	}{
		{"127.0.0.1:1901", true},
		{"localhost:1901", true},
		{"127.0.0.1", true},
		{"[::1]:1901", true},
		{"[::1]", true},
		{"::1", true},
		{"evil.example.com", false},
		{"192.168.1.5:1901", false},
		{"localhost.evil.com", false},
		{"127.0.0.1.evil.com:1901", false},
	}
	for _, tc := range cases {
		if got := mediaLoopbackHostOK(tc.host); got != tc.ok {
			t.Fatalf("mediaLoopbackHostOK(%q) = %v, want %v", tc.host, got, tc.ok)
		}
	}
}

func TestMediaHTTPPanicRecovered(t *testing.T) {
	// A panicking handler must yield 500, not crash the server.
	mux := http.NewServeMux()
	mux.HandleFunc("GET /boom", func(w http.ResponseWriter, r *http.Request) { panic("test explosion") })
	h := mediaHTTPGuard(mux)
	req := httptest.NewRequest("GET", "/boom", nil)
	req.Host = "127.0.0.1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("panic status = %d, want 500", rec.Code)
	}
}

func TestMediaWorkFileFixtureRoundTrip(t *testing.T) {
	// The ID→path mapping serves exactly the managed file bytes.
	mediaJobsTestEnv(t, "ok")
	fixture := pngFixtureBytes(t)
	dir := t.TempDir()
	p := filepath.Join(dir, "managed.png")
	if err := os.WriteFile(p, fixture, 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	// mediaRegisterWork takes mediaWorksMu itself — do not hold it here.
	if err := mediaRegisterWork(&MediaArtifact{ID: "art-x", JobID: "job-x", Type: "image", Path: p, Size: int64(len(fixture))}); err != nil {
		t.Fatalf("register work: %v", err)
	}
	got, err := MediaArtifactPath("art-x")
	if err != nil || got != p {
		t.Fatalf("path = %s/%v", got, err)
	}
	if _, err := MediaArtifactPath("art-missing"); err == nil {
		t.Fatalf("unknown id must not resolve")
	}
}
