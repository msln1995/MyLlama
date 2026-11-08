// The inference arbiter is the single atomic gate between llama-server and
// media generation (plan §5.5). The app's conservative policy: when the
// managed llama-server is alive (or committed to starting), media may not
// run; while media holds the reservation, llama may not start. The policy
// covers the GUI start path, the control plane, process adoption and mode
// switches because every one of them must claim the llama side through this
// arbiter before spawning.
//
// Correctness shape: both sides perform their liveness check AND their flag
// write under one mutex, so there is no "checked then started" window. The
// llama side tracks the starting phase separately (llamaStarting) so a
// spawn in flight cannot lose a race against a media submit; the media side
// holds a single reservation (P1 runs one generation job at a time).
//
// This arbiter only manages the app's own engines — other applications'
// GPU/memory usage is out of scope (plan §5.5 closing note).
package core

import "sync"

// inferenceArbiter holds the two-sided state. All fields are guarded by mu.
// Media reservations are a SET of holders (one per non-terminal job): the
// P1 queue runs one job at a time but several submitted jobs may be waiting,
// and every one of them blocks a llama start (plan §5.5 conservative policy).
type inferenceArbiter struct {
	mu            sync.Mutex
	llamaActive   bool              // a managed llama-server is up (incl. adopted)
	llamaStarting bool              // a spawn/adoption is in flight
	mediaHolders  map[string]string // holder id -> description
}

var mediaArbiter inferenceArbiter

// ArbiterClaimLLama marks a llama spawn/adoption as in flight. Returns false
// when media holds the reservation — the caller must abort the start and
// surface the conflict (plan §5.5: 切换门禁应在拉起继任进程之前生效).
func ArbiterClaimLLama() bool {
	mediaArbiter.mu.Lock()
	defer mediaArbiter.mu.Unlock()
	if len(mediaArbiter.mediaHolders) > 0 {
		return false
	}
	mediaArbiter.llamaStarting = true
	return true
}

// ArbiterLLamaUp marks the llama side as live (spawn succeeded / process
// adopted). Clears the starting flag.
func ArbiterLLamaUp() {
	mediaArbiter.mu.Lock()
	mediaArbiter.llamaStarting = false
	mediaArbiter.llamaActive = true
	mediaArbiter.mu.Unlock()
}

// ArbiterAbandonLLama releases a failed/aborted spawn claim without marking
// the llama side live.
func ArbiterAbandonLLama() {
	mediaArbiter.mu.Lock()
	mediaArbiter.llamaStarting = false
	mediaArbiter.mu.Unlock()
}

// ArbiterLLamaDown clears the live flag (llama-server stopped or its process
// exited). Unknown exit states are intentionally NOT cleared here — the stop
// path decides, see bridge.stopServerInternal.
func ArbiterLLamaDown() {
	mediaArbiter.mu.Lock()
	mediaArbiter.llamaActive = false
	mediaArbiter.mu.Unlock()
}

// ArbiterTryReserveMedia atomically reserves the inference slot for media.
// It fails while the llama side is live or starting, or while another media
// reservation is held. holder is recorded for diagnostics and must be
// non-empty.
func ArbiterTryReserveMedia(holder string) error {
	if holder == "" {
		holder = "unknown"
	}
	mediaArbiter.mu.Lock()
	defer mediaArbiter.mu.Unlock()
	if mediaArbiter.llamaActive || mediaArbiter.llamaStarting {
		return errLLamaBusy{reason: "the chat/API service is running; stop it first (existing requests will be interrupted)"}
	}
	if mediaArbiter.mediaHolders == nil {
		mediaArbiter.mediaHolders = map[string]string{}
	}
	mediaArbiter.mediaHolders[holder] = holder
	return nil
}

// ArbiterReleaseMedia drops the media reservation held by holder. Releasing
// with a mismatched holder is a no-op (defensive against double release).
func ArbiterReleaseMedia(holder string) {
	mediaArbiter.mu.Lock()
	defer mediaArbiter.mu.Unlock()
	if holder == "" {
		mediaArbiter.mediaHolders = nil
		return
	}
	delete(mediaArbiter.mediaHolders, holder)
}

// ArbiterMediaHolders reports the live media reservation holders (diagnostics
// and tests).
func ArbiterMediaHolders() []string {
	mediaArbiter.mu.Lock()
	defer mediaArbiter.mu.Unlock()
	out := make([]string, 0, len(mediaArbiter.mediaHolders))
	for h := range mediaArbiter.mediaHolders {
		out = append(out, h)
	}
	return out
}

// ArbiterMediaBlocked reports whether the llama side currently blocks media.
func ArbiterMediaBlocked() bool {
	mediaArbiter.mu.Lock()
	defer mediaArbiter.mu.Unlock()
	return mediaArbiter.llamaActive || mediaArbiter.llamaStarting
}

// errLLamaBusy is the typed gate failure so callers (GUI and HTTP) can map it
// to a distinct, actionable message.
type errLLamaBusy struct{ reason string }

func (e errLLamaBusy) Error() string { return e.reason }

// IsLLamaBusy reports whether err is the "llama is running" gate failure.
func IsLLamaBusy(err error) bool {
	_, ok := err.(errLLamaBusy)
	return ok
}
