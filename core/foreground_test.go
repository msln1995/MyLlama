package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// ─── Android foreground service wiring ───────────────────────────

// keepAwakeRecorder captures the seam calls (the wait goroutine writes
// concurrently with the test's reads, hence the mutex).
type keepAwakeRecorder struct {
	mu       sync.Mutex
	titles   []string
	texts    []string
	releases int
}

func (r *keepAwakeRecorder) on(title, text string) {
	r.mu.Lock()
	r.titles = append(r.titles, title)
	r.texts = append(r.texts, text)
	r.mu.Unlock()
}

func (r *keepAwakeRecorder) off() {
	r.mu.Lock()
	r.releases++
	r.mu.Unlock()
}

// counts returns the number of raises, the latest notification body and the
// number of releases.
func (r *keepAwakeRecorder) counts() (raises int, lastText string, releases int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	lastText = ""
	if len(r.texts) > 0 {
		lastText = r.texts[len(r.texts)-1]
	}
	return len(r.texts), lastText, r.releases
}

// recordKeepAwake swaps the keepAwake seams for the recorder, restoring the
// production bindings on cleanup.
func recordKeepAwake(t *testing.T) *keepAwakeRecorder {
	t.Helper()
	rec := &keepAwakeRecorder{}
	origOn, origOff := keepAwakeOnFn, keepAwakeOffFn
	t.Cleanup(func() { keepAwakeOnFn, keepAwakeOffFn = origOn, origOff })
	keepAwakeOnFn = rec.on
	keepAwakeOffFn = rec.off
	return rec
}

// withTempServerLogFile points serverLogFile at a temp path so the spawn path's
// log-file capture never writes into the repository.
func withTempServerLogFile(t *testing.T) {
	t.Helper()
	orig := serverLogFile
	serverLogFile = filepath.Join(t.TempDir(), "server.log")
	t.Cleanup(func() { serverLogFile = orig })
}

// TestSpawnServerProcessRaisesKeepAwakeForChildLifetime verifies the server
// lifecycle owns the foreground service: a real spawn raises it with the
// resident model in the notification body, and the normal stop path releases it
// once the child exits.
func TestSpawnServerProcessRaisesKeepAwakeForChildLifetime(t *testing.T) {
	saveServerState(t)
	withPlatformGOOS(t, "linux") // buildChildEnv stays nil: the child inherits the test env
	withLanguage(t, "en")
	withTempServerLogFile(t)
	rec := recordKeepAwake(t)

	t.Setenv("GO_WANT_HELPER_PROCESS", "1")
	if err := spawnServerProcess(os.Args[0], []string{"-test.run=^TestHelperProcess$"},
		ServerConfig{Port: 8080}, "qwen-test"); err != nil {
		t.Fatalf("spawnServerProcess: %v", err)
	}
	raises, lastText, releases := rec.counts()
	if raises != 1 {
		t.Fatalf("keepAwakeOn calls = %d, want 1", raises)
	}
	if lastText != "qwen-test" {
		t.Errorf("notification body = %q, want the resident model", lastText)
	}
	if releases != 0 {
		t.Fatalf("keepAwakeOff called while the child still runs (%d)", releases)
	}

	if err := stopServerInternal(); err != nil {
		t.Fatalf("stopServerInternal: %v", err)
	}
	// stopServerInternal returns only once the wait goroutine closed done, and
	// the release happens before that close, so the count is final here.
	if _, _, releases := rec.counts(); releases != 1 {
		t.Errorf("keepAwakeOff calls after stop = %d, want 1", releases)
	}
}

// TestOutgoingChildKeepsKeepAwakeAcrossRestart verifies the restart guard: with
// a second child already current, the outgoing instance's wait goroutine must
// not release the foreground service (a model switch would otherwise drop the
// app back to the frozen background set mid-restart).
func TestOutgoingChildKeepsKeepAwakeAcrossRestart(t *testing.T) {
	saveServerState(t)
	withPlatformGOOS(t, "linux")
	withLanguage(t, "en")
	withTempServerLogFile(t)
	rec := recordKeepAwake(t)

	t.Setenv("GO_WANT_HELPER_PROCESS", "1")
	if err := spawnServerProcess(os.Args[0], []string{"-test.run=^TestHelperProcess$"},
		ServerConfig{Port: 8080}, "first"); err != nil {
		t.Fatalf("first spawn: %v", err)
	}
	serverMu.Lock()
	first := serverCmd
	serverMu.Unlock()

	// Replace the current server the way a direct-mode restart does.
	if err := spawnServerProcess(os.Args[0], []string{"-test.run=^TestHelperProcess$"},
		ServerConfig{Port: 8080}, "second"); err != nil {
		t.Fatalf("second spawn: %v", err)
	}
	serverMu.Lock()
	current := serverCmd
	serverMu.Unlock()
	if current == first {
		t.Fatal("second spawn did not become the current server")
	}

	first.Process.Kill()
	first.Wait()
	// Give the outgoing wait goroutine a window to run before asserting.
	time.Sleep(200 * time.Millisecond)
	if _, _, releases := rec.counts(); releases != 0 {
		t.Fatalf("outgoing child released the foreground service (%d calls)", releases)
	}

	// The current child exits the same way and must release it.
	current.Process.Kill()
	current.Wait()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, _, releases := rec.counts(); releases == 1 {
			return
		}
		if time.Now().After(deadline) {
			_, _, releases := rec.counts()
			t.Fatalf("keepAwakeOff calls after the current child exited = %d, want 1", releases)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestServingNotificationNamesTheResidentModel pins the notification copy:
// direct mode shows the resident model, an empty resident falls back to the
// language-appropriate generic body.
func TestServingNotificationNamesTheResidentModel(t *testing.T) {
	withLanguage(t, "en")
	title, text := servingNotification("qwen3.5-4b")
	if title != "MyLlama is running an inference service" {
		t.Errorf("title = %q", title)
	}
	if text != "qwen3.5-4b" {
		t.Errorf("text with resident model = %q, want the model name", text)
	}
	if _, text := servingNotification(""); text != "llama-server is running in the background" {
		t.Errorf("text without a resident model = %q", text)
	}

	withLanguage(t, "zh")
	if title, _ := servingNotification("qwen3.5-4b"); title != "MyLlama 正在运行推理服务" {
		t.Errorf("zh title = %q", title)
	}
	if _, text := servingNotification(""); text != "llama-server 正在后台运行" {
		t.Errorf("zh fallback text = %q", text)
	}
}

// TestForegroundPayloadIsValidBridgeJSON verifies the {"title","text"} object
// the Java bridge parses: quotes, backslashes and non-ASCII must survive JSON
// round-tripping.
func TestForegroundPayloadIsValidBridgeJSON(t *testing.T) {
	payload := foregroundPayload(`My "title" \ path`, "模型 «qwen»")
	var got struct {
		Title string `json:"title"`
		Text  string `json:"text"`
	}
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("payload %q is not valid JSON: %v", payload, err)
	}
	if got.Title != `My "title" \ path` || got.Text != "模型 «qwen»" {
		t.Errorf("round-trip = %+v", got)
	}
}
