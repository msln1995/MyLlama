// Media generation job management: the serial task queue, its persisted
// state machine, idempotent submission, cancellation with bounded exit
// waits, artifact validation and the works registry (plan §5.4 / P1
// checklist). Both the GUI bindings and the local HTTP service submit
// through this one core — there is exactly one task state machine.
//
// Success is defined by the plan's bar: the engine process must exit
// successfully AND the expected artifact must exist AND pass format checks.
// Anything else is a failure — a zero-exit process with no valid file is a
// failure ("fake success" defense), and intermediate files never become
// works.
package core

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Media job statuses. The happy path is queued → preparing → running →
// finalizing → succeeded. Failure enters failed; cancellation enters
// cancelling and lands on cancelled only after the process exit is confirmed;
// a restart recovers in-flight jobs to interrupted (the user re-runs them —
// P1 never auto-resumes sampling).
const (
	MediaJobQueued      = "queued"
	MediaJobPreparing   = "preparing"
	MediaJobRunning     = "running"
	MediaJobFinalizing  = "finalizing"
	MediaJobSucceeded   = "succeeded"
	MediaJobFailed      = "failed"
	MediaJobCancelling  = "cancelling"
	MediaJobCancelled   = "cancelled"
	MediaJobInterrupted = "interrupted"
)

// mediaJobTerminal reports whether a status is final.
func mediaJobTerminal(s string) bool {
	switch s {
	case MediaJobSucceeded, MediaJobFailed, MediaJobCancelled, MediaJobInterrupted:
		return true
	}
	return false
}

// mediaJobTimeout is the hard per-job engine timeout (plan §5.4: 可解释的
// 任务超时). Generations are minutes-scale today; hours-scale video comes
// with its own budget later.
var mediaJobTimeout = 90 * time.Minute

// mediaJobQueueLimit bounds the number of queued (not yet running) jobs.
const mediaJobQueueLimit = 10

// mediaJobLogLimit caps the persisted per-job log tail.
const mediaJobLogLimit = 200

// MediaArtifact is a validated generated work.
type MediaArtifact struct {
	ID        string         `json:"id"`
	JobID     string         `json:"jobId"`
	Type      string         `json:"type"` // "image" (P1); "video" later
	Path      string         `json:"path"` // absolute, machine-local (paths are machine-local state)
	Size      int64          `json:"size"`
	SHA256    string         `json:"sha256"`
	Width     int            `json:"width"`
	Height    int            `json:"height"`
	CreatedAt time.Time      `json:"createdAt"`
	Params    MediaJobParams `json:"params"` // parameter provenance (plan §5.6)
}

// MediaJob is one generation task. Persisted fields only — logs are capped
// tails, no ephemeral handles.
type MediaJob struct {
	ID            string         `json:"id"`
	RequestID     string         `json:"requestId,omitempty"`
	ModelID       string         `json:"modelId"`
	VariantID     string         `json:"variantId"`
	Params        MediaJobParams `json:"params"`
	Status        string         `json:"status"`
	Error         string         `json:"error,omitempty"`
	EngineVersion string         `json:"engineVersion,omitempty"`
	EngineCommit  string         `json:"engineCommit,omitempty"`
	CreatedAt     time.Time      `json:"createdAt"`
	StartedAt     time.Time      `json:"startedAt,omitempty"`
	FinishedAt    time.Time      `json:"finishedAt,omitempty"`
	LogTail       []string       `json:"logTail,omitempty"`
	Artifact      *MediaArtifact `json:"artifact,omitempty"`
}

// mediaJobsFile is the persisted jobs state (bounded history).
type mediaJobsFile struct {
	SchemaVersion int        `json:"schemaVersion"`
	Jobs          []MediaJob `json:"jobs"`
}

const mediaJobsSchemaVersion = 1

// mediaJobsHistory is how many jobs the state file keeps (newest first).
const mediaJobsHistory = 100

var (
	mediaJobsPath = mediaJobsFilePath // test seam
	// mediaRandID draws job/artifact ids; tests can pin it.
	mediaRandID = func() string {
		var b [8]byte
		if _, err := rand.Read(b[:]); err != nil {
			return fmt.Sprintf("j%d", time.Now().UnixNano())
		}
		return hex.EncodeToString(b[:])
	}
)

// mediaJobManager is the single task core. mu guards everything; runCh is
// the serial queue consumed by the one worker goroutine.
type mediaJobManager struct {
	mu        sync.Mutex
	jobs      []*MediaJob // newest first, mirrored into the state file
	cancelFn  context.CancelFunc
	consumer  chan *MediaJob
	consumeMu sync.Once
	stopCh    chan struct{}
	stopped   bool
}

var mediaJobsMgr = &mediaJobManager{}

// loadMediaJobs reads the persisted history and recovers statuses: anything
// non-terminal at load time is interrupted (plan §5.4: 应用意外退出后的活动
// 任务恢复为 interrupted；排队任务重启后要求用户显式重试).
func loadMediaJobs() []*MediaJob {
	data, err := os.ReadFile(mediaJobsPath())
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[WARN] Cannot read media jobs file, starting empty: %v", err)
		}
		return nil
	}
	var f mediaJobsFile
	if err := json.Unmarshal(data, &f); err != nil {
		log.Printf("[WARN] Cannot parse media jobs file, starting empty: %v", err)
		return nil
	}
	if f.SchemaVersion != mediaJobsSchemaVersion {
		log.Printf("[WARN] Media jobs schema version %d unsupported, starting empty", f.SchemaVersion)
		return nil
	}
	recovered := 0
	for i := range f.Jobs {
		if !mediaJobTerminal(f.Jobs[i].Status) {
			f.Jobs[i].Status = MediaJobInterrupted
			f.Jobs[i].Error = "app restarted while the job was in flight"
			f.Jobs[i].FinishedAt = time.Now()
			recovered++
		}
	}
	if recovered > 0 {
		log.Printf("[INFO] Media job recovery: %d in-flight job(s) marked interrupted", recovered)
		if err := saveMediaJobsFile(&mediaJobsFile{SchemaVersion: mediaJobsSchemaVersion, Jobs: f.Jobs}); err != nil {
			log.Printf("[WARN] Cannot persist recovered media jobs: %v", err)
		}
	}
	out := make([]*MediaJob, len(f.Jobs))
	for i := range f.Jobs {
		j := f.Jobs[i]
		out[i] = &j
	}
	return out
}

// saveMediaJobsLocked snapshots the in-memory jobs into the state file
// (bounded, newest first). Caller holds mediaJobsMgr.mu.
func saveMediaJobsLocked() {
	out := make([]MediaJob, 0, len(mediaJobsMgr.jobs))
	for _, j := range mediaJobsMgr.jobs {
		out = append(out, *j)
		if len(out) >= mediaJobsHistory {
			break
		}
	}
	if err := saveMediaJobsFile(&mediaJobsFile{SchemaVersion: mediaJobsSchemaVersion, Jobs: out}); err != nil {
		log.Printf("[WARN] Cannot persist media jobs: %v", err)
	}
}

func saveMediaJobsFile(f *mediaJobsFile) error {
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal media jobs: %w", err)
	}
	return atomicWriteFile(mediaJobsPath(), data, 0644)
}

// mediaJobsStart ensures the consumer goroutine is running and seeded with
// recovered history. Called from App startup and lazily from StartMediaJob.
func mediaJobsStart() {
	mediaJobsMgr.consumeMu.Do(func() {
		mediaJobsMgr.jobs = loadMediaJobs()
		mediaJobsMgr.consumer = make(chan *MediaJob, mediaJobQueueLimit)
		mediaJobsMgr.stopCh = make(chan struct{})
		go mediaJobConsumer(mediaJobsMgr, mediaJobsMgr.consumer, mediaJobsMgr.stopCh)
		log.Println("[INFO] Media job manager started")
	})
}

// mediaJobConsumer is the single worker: one job at a time, in submission
// order (plan §5.4: 默认单任务串行). It captures its own manager so a test
// that swaps the global manager cannot have an old consumer mutate the new
// one's state.
func mediaJobConsumer(mgr *mediaJobManager, ch <-chan *MediaJob, stop <-chan struct{}) {
	for {
		select {
		case j := <-ch:
			mediaRunOneJob(mgr, j)
		case <-stop:
			return
		}
	}
}

// appendJobLog appends one engine log line to the job's capped tail.
// Caller holds the manager lock.
func appendJobLog(j *MediaJob, line string) {
	j.LogTail = append(j.LogTail, line)
	if len(j.LogTail) > mediaJobLogLimit {
		j.LogTail = j.LogTail[len(j.LogTail)-mediaJobLogLimit:]
	}
}

// mediaRunOneJob drives one job through its phases. It assumes the job was
// already validated and enqueued by StartMediaJob.
func mediaRunOneJob(mgr *mediaJobManager, j *MediaJob) {
	mgr.mu.Lock()
	if j.Status == MediaJobCancelled {
		// Cancelled while queued: nothing to run; drop the reservation that
		// was taken at submit time.
		mgr.mu.Unlock()
		ArbiterReleaseMedia(j.ID)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	mgr.cancelFn = cancel
	setJobStatus := func(status, errMsg string) {
		j.Status = status
		j.Error = errMsg
		if mediaJobTerminal(status) {
			j.FinishedAt = time.Now()
		}
		saveMediaJobsLocked()
	}
	setJobStatus(MediaJobPreparing, "")
	j.StartedAt = time.Now()
	mgr.mu.Unlock()

	finish := func(status, errMsg string) {
		cancel()
		// Release BEFORE the terminal write: once the status is terminal, the
		// reservation must already be gone (shutdown/cleanup rely on that).
		ArbiterReleaseMedia(j.ID)
		mgr.mu.Lock()
		setJobStatus(status, errMsg)
		mgr.cancelFn = nil
		mgr.mu.Unlock()
	}

	// Phase gates (after queued): engine, install, args, output dir.
	eng := MediaRuntimeStatusNow()
	if err := mediaValidateEngineForProfile(eng); err != nil {
		finish(MediaJobFailed, err.Error())
		return
	}
	mgr.mu.Lock()
	j.EngineVersion, j.EngineCommit = eng.Version, eng.Commit
	mgr.mu.Unlock()

	status, comps, err := MediaInstallStatus(j.ModelID, j.VariantID)
	if err != nil {
		finish(MediaJobFailed, err.Error())
		return
	}
	if status != "ready" {
		finish(MediaJobFailed, fmt.Sprintf("model package %s/%s is not ready (status %s); (re)import its components first", j.ModelID, j.VariantID, status))
		return
	}
	_, variant, err := mediaVariantLookup(j.ModelID, j.VariantID)
	if err != nil {
		finish(MediaJobFailed, err.Error())
		return
	}
	install := &MediaInstallRecord{ModelID: j.ModelID, VariantID: j.VariantID, Components: comps}
	outDir := filepath.Join(activeMediaOutputsDir(), "job-"+j.ID)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		finish(MediaJobFailed, fmt.Sprintf("create output dir: %v", err))
		return
	}
	outPath := filepath.Join(outDir, "output.png")
	args, err := buildMediaImageArgs(variant, install, j.Params, outPath)
	if err != nil {
		finish(MediaJobFailed, err.Error())
		return
	}

	mgr.mu.Lock()
	setJobStatus(MediaJobRunning, "")
	mgr.mu.Unlock()

	// The engine runs with the job id as the arbiter holder; the reservation
	// was taken by StartMediaJob (atomic with its llama check).
	res := mediaExecRun(ctx, eng.EnginePath, args, mediaJobTimeout, func(line string) {
		mgr.mu.Lock()
		appendJobLog(j, line)
		mgr.mu.Unlock()
	})

	mgr.mu.Lock()
	if ctx.Err() == context.Canceled {
		setJobStatus(MediaJobCancelled, "cancelled by user")
		mgr.mu.Unlock()
		return
	}
	setJobStatus(MediaJobFinalizing, "")
	mgr.mu.Unlock()

	// Finalizing: validate the artifact. A zero exit without a valid file is
	// a failure, never a success (plan §5.4 fake-success defense).
	if res.TimedOut {
		finish(MediaJobFailed, res.Err.Error())
		return
	}
	if res.ExitCode != 0 {
		finish(MediaJobFailed, fmt.Sprintf("engine exited with code %d: %v", res.ExitCode, res.Err))
		return
	}
	art, err := mediaValidateImageArtifact(outPath, j)
	if err != nil {
		finish(MediaJobFailed, fmt.Sprintf("artifact validation failed: %v", err))
		return
	}
	mgr.mu.Lock()
	j.Artifact = art
	mgr.mu.Unlock()
	// finish() takes the manager lock itself and releases the arbiter
	// reservation on this terminal transition.
	finish(MediaJobSucceeded, "")
	if err := mediaRegisterWork(art); err != nil {
		// The file is fine and the job succeeded; a failed works write must
		// not fake-fail the job, but it must be loud.
		log.Printf("[ERROR] Cannot register media work: %v", err)
	}
}

// mediaValidateImageArtifact validates the generated image: exists, PNG
// format with parseable dimensions, non-trivial size; then hashes it.
func mediaValidateImageArtifact(path string, j *MediaJob) (*MediaArtifact, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("output file missing: %w", err)
	}
	if fi.Size() < 64 {
		return nil, fmt.Errorf("output file suspiciously small (%d bytes)", fi.Size())
	}
	w, h, err := parsePNGSize(path)
	if err != nil {
		return nil, err
	}
	sum, err := fileSHA256(path)
	if err != nil {
		return nil, err
	}
	return &MediaArtifact{
		ID:        "art-" + j.ID,
		JobID:     j.ID,
		Type:      "image",
		Path:      path,
		Size:      fi.Size(),
		SHA256:    sum,
		Width:     w,
		Height:    h,
		CreatedAt: time.Now(),
		Params:    j.Params,
	}, nil
}

// parsePNGSize reads the PNG signature and IHDR dimensions. Pure and
// dependency-free: 8-byte signature, then IHDR length+type, then 4-byte
// big-endian width and height.
func parsePNGSize(path string) (int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	defer f.Close()
	head := make([]byte, 24)
	if _, err := ioReadFull(f, head); err != nil {
		return 0, 0, fmt.Errorf("short read on PNG header: %w", err)
	}
	pngSig := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}
	for i, b := range pngSig {
		if head[i] != b {
			return 0, 0, fmt.Errorf("%s is not a valid PNG (bad signature)", filepath.Base(path))
		}
	}
	if head[12] != 'I' || head[13] != 'H' || head[14] != 'D' || head[15] != 'R' {
		return 0, 0, fmt.Errorf("%s is not a valid PNG (IHDR not first)", filepath.Base(path))
	}
	w := int(binary.BigEndian.Uint32(head[16:20]))
	h := int(binary.BigEndian.Uint32(head[20:24]))
	if w <= 0 || h <= 0 || w > 65535 || h > 65535 {
		return 0, 0, fmt.Errorf("%s has implausible dimensions %dx%d", filepath.Base(path), w, h)
	}
	return w, h, nil
}

// ioReadFull reads exactly len(buf) bytes (local io.ReadFull alias to keep
// imports minimal in this file).
func ioReadFull(f *os.File, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := f.Read(buf[total:])
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

// works registry ----------------------------------------------------------

// mediaWorksFile is the persisted works registry (validated artifacts only).
type mediaWorksFile struct {
	SchemaVersion int             `json:"schemaVersion"`
	Works         []MediaArtifact `json:"works"`
}

const mediaWorksSchemaVersion = 1

var mediaWorksPath = mediaWorksFilePath // test seam
var mediaWorksMu sync.Mutex

// mediaRegisterWork appends the artifact to the works registry.
func mediaRegisterWork(art *MediaArtifact) error {
	mediaWorksMu.Lock()
	defer mediaWorksMu.Unlock()
	f := mediaLoadWorks()
	f.Works = append(f.Works, *art)
	if len(f.Works) > mediaJobsHistory {
		f.Works = f.Works[len(f.Works)-mediaJobsHistory:]
	}
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal media works: %w", err)
	}
	return atomicWriteFile(mediaWorksPath(), data, 0644)
}

func mediaLoadWorks() *mediaWorksFile {
	data, err := os.ReadFile(mediaWorksPath())
	if err == nil {
		var f mediaWorksFile
		if json.Unmarshal(data, &f) == nil && f.SchemaVersion == mediaWorksSchemaVersion {
			return &f
		}
	}
	return &mediaWorksFile{SchemaVersion: mediaWorksSchemaVersion}
}

// ListMediaArtifacts returns the validated works, newest first.
func ListMediaArtifacts() []MediaArtifact {
	mediaWorksMu.Lock()
	defer mediaWorksMu.Unlock()
	f := mediaLoadWorks()
	out := make([]MediaArtifact, len(f.Works))
	for i := range f.Works {
		out[i] = f.Works[len(f.Works)-1-i]
	}
	return out
}

// MediaArtifactPath resolves a work id to its local file path (the HTTP
// artifact endpoint maps this id → managed file; nothing exposes arbitrary
// disk paths, plan §5.6).
func MediaArtifactPath(id string) (string, error) {
	mediaWorksMu.Lock()
	defer mediaWorksMu.Unlock()
	f := mediaLoadWorks()
	for i := range f.Works {
		if f.Works[i].ID == id {
			return f.Works[i].Path, nil
		}
	}
	return "", fmt.Errorf("media work %q not found", id)
}

// job lifecycle API -------------------------------------------------------

// MediaJobStartRequest is the submission payload (GUI binding + HTTP share
// it). RequestID enables idempotent retries.
type MediaJobStartRequest struct {
	RequestID string         `json:"requestId"`
	ModelID   string         `json:"modelId"`
	VariantID string         `json:"variantId"`
	Params    MediaJobParams `json:"params"`
}

// StartMediaJob validates and enqueues one generation job. Idempotent by
// RequestID: a resubmission returns the original job (plan §5.4: 一个任务仅
// 启动一次，重复请求返回原任务). The llama-vs-media gate is taken atomically
// here; the reservation is released when the job reaches a terminal state.
func StartMediaJob(req MediaJobStartRequest) (*MediaJob, error) {
	mediaJobsStart() // lazy ensure (also covers tests before app startup)

	if req.ModelID == "" || req.VariantID == "" {
		return nil, fmt.Errorf("modelId and variantId are required")
	}
	_, v, err := mediaVariantLookup(req.ModelID, req.VariantID)
	if err != nil {
		return nil, err
	}
	if v.Profile.Kind != "image" {
		return nil, fmt.Errorf("profile kind %q is not open for jobs yet", v.Profile.Kind)
	}
	params := req.Params
	if params.Seed < 0 {
		var b [8]byte
		_, _ = rand.Read(b[:])
		params.Seed = int64(binary.BigEndian.Uint64(b[:]) & 0x7fffffffffffffff)
	}
	if err := validateMediaJobParams(params, v.Profile); err != nil {
		return nil, fmt.Errorf("invalid job parameters: %w", err)
	}

	mgr := mediaJobsMgr
	mgr.mu.Lock()
	if req.RequestID != "" {
		for _, j := range mgr.jobs {
			if j.RequestID == req.RequestID {
				snap := *j
				mgr.mu.Unlock()
				return &snap, nil
			}
		}
	}
	// Queue bound: reject when the serial queue is full (plan §5.4).
	queued := 0
	for _, j := range mgr.jobs {
		if j.Status == MediaJobQueued {
			queued++
		}
	}
	if queued >= mediaJobQueueLimit {
		mgr.mu.Unlock()
		return nil, fmt.Errorf("media job queue is full (%d); try again later", mediaJobQueueLimit)
	}
	mgr.mu.Unlock()

	// The llama-vs-media gate is atomic but must be taken WITHOUT holding the
	// manager lock (the stop path takes arbiter + manager in the other order).
	// The job id doubles as the reservation holder so every terminal path
	// (success, failure, cancel, queue-cancel) releases by it.
	job := &MediaJob{
		ID:        "job-" + mediaRandID(),
		RequestID: req.RequestID,
		ModelID:   req.ModelID,
		VariantID: req.VariantID,
		Params:    params,
		Status:    MediaJobQueued,
		CreatedAt: time.Now(),
	}
	if err := ArbiterTryReserveMedia(job.ID); err != nil {
		return nil, err
	}
	// From here the reservation must be released on every exit path.
	release := func() { ArbiterReleaseMedia(job.ID) }
	mgr.mu.Lock()
	// Re-check idempotency under the lock (a racing duplicate submit).
	if req.RequestID != "" {
		for _, j2 := range mgr.jobs {
			if j2.RequestID == req.RequestID {
				snap := *j2
				mgr.mu.Unlock()
				release()
				return &snap, nil
			}
		}
	}
	mgr.jobs = append([]*MediaJob{job}, mgr.jobs...)
	saveMediaJobsLocked()
	mgr.mu.Unlock()

	select {
	case mgr.consumer <- job:
	default:
		// The consumer channel is bounded by the same limit; a full channel
		// means the queued-count check raced. Back out cleanly.
		mgr.mu.Lock()
		job.Status = MediaJobFailed
		job.Error = "queue overflow"
		job.FinishedAt = time.Now()
		saveMediaJobsLocked()
		mgr.mu.Unlock()
		release()
		return nil, fmt.Errorf("media job queue overflow")
	}
	// Snapshot under the lock: the consumer may already be running the job
	// and writing its status fields (the returned snapshot must be coherent).
	mgr.mu.Lock()
	snap := *job
	mgr.mu.Unlock()
	return &snap, nil
}

// GetMediaJobs returns a snapshot of all jobs, newest first.
func GetMediaJobs() []MediaJob {
	mediaJobsStart()
	mediaJobsMgr.mu.Lock()
	defer mediaJobsMgr.mu.Unlock()
	out := make([]MediaJob, len(mediaJobsMgr.jobs))
	for i, j := range mediaJobsMgr.jobs {
		out[i] = *j
	}
	return out
}

// CancelMediaJob cancels a job. Queued jobs are cancelled immediately; a
// running job moves to cancelling and lands on cancelled only after the
// engine exit is confirmed (the consumer does the transition). Terminal jobs
// reject cancellation.
func CancelMediaJob(jobID string) error {
	mediaJobsStart()
	mgr := mediaJobsMgr
	mgr.mu.Lock()
	defer mgr.mu.Unlock()
	for _, j := range mgr.jobs {
		if j.ID != jobID {
			continue
		}
		switch j.Status {
		case MediaJobQueued:
			j.Status = MediaJobCancelled
			j.Error = "cancelled while queued"
			j.FinishedAt = time.Now()
			saveMediaJobsLocked()
			return nil
		case MediaJobPreparing, MediaJobRunning:
			j.Status = MediaJobCancelling
			saveMediaJobsLocked()
			if mgr.cancelFn != nil {
				mgr.cancelFn() // graceful-first termination happens in the runner
			}
			return nil
		case MediaJobCancelling:
			return nil // already in flight
		case MediaJobFinalizing:
			// Past the engine exit: the job resolves to succeeded/failed
			// momentarily. Rejecting keeps the state machine from getting
			// stuck between cancelling and a terminal status.
			return fmt.Errorf("job %s is finalizing; cancel came too late", jobID)
		default:
			return fmt.Errorf("job %s already finished (%s)", jobID, j.Status)
		}
	}
	return fmt.Errorf("media job %q not found", jobID)
}

// mediaJobsShutdown cancels any active job and waits, bounded, for the
// terminal transition (plan §5.5: 正常退出取消媒体并有界等待). Called from the
// app shutdown path.
func mediaJobsShutdown() {
	mgr := mediaJobsMgr
	mgr.mu.Lock()
	if mgr.stopped {
		mgr.mu.Unlock()
		return
	}
	mgr.stopped = true
	cancel := mgr.cancelFn
	// Queued jobs will never run after the consumer stops: cancel them and
	// drop their reservations (plan §5.5: 正常退出取消媒体).
	for _, j := range mgr.jobs {
		if j.Status == MediaJobQueued {
			j.Status = MediaJobCancelled
			j.Error = "cancelled by app shutdown"
			j.FinishedAt = time.Now()
			ArbiterReleaseMedia(j.ID)
		}
	}
	saveMediaJobsLocked()
	mgr.mu.Unlock()
	if closeStop := mgr.stopCh; closeStop != nil {
		close(closeStop)
	}
	if cancel != nil {
		cancel()
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		mgr.mu.Lock()
		done := true
		for _, j := range mgr.jobs {
			if !mediaJobTerminal(j.Status) {
				done = false
				break
			}
		}
		mgr.mu.Unlock()
		if done {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	log.Println("[WARN] Media job shutdown wait timed out; the engine process may still be exiting")
}
