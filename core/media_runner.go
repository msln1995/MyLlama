// Media job execution: command construction (argv array, never shell) and
// the controlled child-process run (logs, cancel, bounded exit wait,
// timeout). The runner is profile-family aware: it maps component roles to
// engine flags for the families it knows and fails closed on unknown ones
// (plan §5.4: 以参数数组启动已验证的引擎).
package core

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// MediaJobParams is the user-visible generation request for an image job.
// Every field is validated against the variant's profile before a process is
// started (plan §5.2: 模型族约束在后端校验，不能只依赖控件范围).
type MediaJobParams struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negativePrompt,omitempty"`
	Seed           int64   `json:"seed"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	CFGScale       float64 `json:"cfgScale"`
	Sampler        string  `json:"sampler,omitempty"`
}

// MediaJobLimits are the hard validation bounds applied to every request
// (profile bounds are intersected with these).
const (
	mediaMaxPromptRunes = 4000
	mediaMaxSteps       = 150
	mediaMaxCFG         = 30.0
)

// validateMediaJobParams checks the request against the profile: prompt
// presence and length, size bounds and alignment, steps and CFG ranges. The
// seed is any int64 (a negative seed is normalized by the caller to a
// random one at run time — the params recorded on the job are the resolved
// values, so repeats are reproducible).
func validateMediaJobParams(p MediaJobParams, profile MediaRunnerProfile) error {
	if strings.TrimSpace(p.Prompt) == "" {
		return fmt.Errorf("prompt is required")
	}
	if len([]rune(p.Prompt)) > mediaMaxPromptRunes {
		return fmt.Errorf("prompt exceeds %d characters", mediaMaxPromptRunes)
	}
	min := profile.MinSize
	if min <= 0 {
		min = 64
	}
	max := profile.MaxSize
	if max <= 0 {
		max = 4096
	}
	mult := profile.SizeMultiple
	if mult <= 0 {
		mult = 8
	}
	if p.Width < min || p.Width > max || p.Height < min || p.Height > max {
		return fmt.Errorf("size %dx%d outside supported range %d..%d per side", p.Width, p.Height, min, max)
	}
	if p.Width%mult != 0 || p.Height%mult != 0 {
		return fmt.Errorf("size %dx%d must be a multiple of %d", p.Width, p.Height, mult)
	}
	if p.Steps < 1 || p.Steps > mediaMaxSteps {
		return fmt.Errorf("steps %d outside 1..%d", p.Steps, mediaMaxSteps)
	}
	if p.CFGScale < 0 || p.CFGScale > mediaMaxCFG {
		return fmt.Errorf("cfg scale %g outside 0..%g", p.CFGScale, mediaMaxCFG)
	}
	return nil
}

// mediaRoleFlag maps component roles to sd-cli flags for the
// z-image-turbo family: diffusion model, Qwen-family LLM text encoder and
// standalone VAE. The mapping fails closed for roles the family cannot wire
// (unknown roles are rejected by the catalog already; this catches families
// wired to roles they do not understand).
func mediaRoleFlag(family, role string) (string, error) {
	switch family {
	case "z-image-turbo":
		switch role {
		case MediaRoleDiffusion:
			return "--diffusion-model", nil
		case MediaRoleTextEncoder:
			return "--llm", nil
		case MediaRoleVAE:
			return "--vae", nil
		}
	}
	return "", fmt.Errorf("runner family %q cannot wire component role %q", family, role)
}

// buildMediaImageArgs constructs the full argv for an image job: component
// wiring from the install record, then the generation parameters, then the
// controlled output path. Pure function — everything it needs is passed in,
// which keeps the runner testable without processes.
func buildMediaImageArgs(v *MediaVariant, install *MediaInstallRecord, params MediaJobParams, outputPath string) ([]string, error) {
	if v.Profile.Kind != "image" {
		return nil, fmt.Errorf("profile kind %q is not runnable as an image job", v.Profile.Kind)
	}
	args := make([]string, 0, 24)
	args = append(args, "-M", "img_gen")
	byRole := map[string]MediaComponentSource{}
	for _, c := range install.Components {
		byRole[c.Role] = c
	}
	for _, comp := range v.Components {
		src, ok := byRole[comp.Role]
		if !ok {
			return nil, fmt.Errorf("install record misses component role %q", comp.Role)
		}
		flag, err := mediaRoleFlag(v.Profile.Family, comp.Role)
		if err != nil {
			return nil, err
		}
		args = append(args, flag, src.Path)
	}
	args = append(args,
		"-p", params.Prompt,
		"-H", mediaArgInt(params.Height),
		"-W", mediaArgInt(params.Width),
		"--steps", mediaArgInt(params.Steps),
		"--cfg-scale", fmt.Sprintf("%g", params.CFGScale),
		"--seed", fmt.Sprintf("%d", params.Seed),
		"-o", outputPath,
	)
	sampler := params.Sampler
	if sampler == "" {
		sampler = v.Profile.Sampler
	}
	if sampler != "" {
		args = append(args, "--sampling-method", sampler)
	}
	if strings.TrimSpace(params.NegativePrompt) != "" {
		args = append(args, "-n", params.NegativePrompt)
	}
	return args, nil
}

// mediaRunResult captures the outcome of one child run.
type mediaRunResult struct {
	ExitCode int
	TimedOut bool
	Err      error
}

// mediaExecRun runs the engine with argv, streaming stdout/stderr lines into
// the sink. Termination is graceful-first on both cancel and timeout:
// interrupt (SIGTERM on Unix), bounded wait (mediaKillGrace), then kill —
// matching plan §5.4 (有界退出等待). hideWindow prevents console flashes on
// Windows (repo convention).
func mediaExecRun(ctx context.Context, enginePath string, args []string, timeout time.Duration, lineSink func(string)) mediaRunResult {
	cmd := exec.Command(enginePath, args...)
	hideWindow(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return mediaRunResult{Err: fmt.Errorf("stdout pipe: %w", err)}
	}
	cmd.Stderr = cmd.Stdout // single merged stream, assembled line-wise
	if err := cmd.Start(); err != nil {
		return mediaRunResult{Err: fmt.Errorf("start %s: %w", filepath.Base(enginePath), err)}
	}
	// One line pump; it ends when the pipes close, which follows Wait.
	pumpDone := make(chan struct{})
	go func() {
		defer close(pumpDone)
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			if lineSink != nil {
				lineSink(sc.Text())
			}
		}
	}()
	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	var waitErr error
	timedOut := false
	terminate := func() {
		_ = cmd.Process.Signal(os.Interrupt)
		select {
		case waitErr = <-waitCh:
		case <-time.After(mediaKillGrace):
			_ = cmd.Process.Kill()
			waitErr = <-waitCh
		}
	}
	select {
	case <-ctx.Done():
		terminate()
	case <-timer.C:
		timedOut = true
		terminate()
	case waitErr = <-waitCh:
	}
	<-pumpDone
	if timedOut {
		return mediaRunResult{ExitCode: cmd.ProcessState.ExitCode(), TimedOut: true, Err: fmt.Errorf("engine timed out after %s", timeout)}
	}
	if ctx.Err() == context.Canceled {
		return mediaRunResult{ExitCode: cmd.ProcessState.ExitCode(), Err: fmt.Errorf("engine stopped by cancel")}
	}
	if waitErr != nil {
		return mediaRunResult{ExitCode: cmd.ProcessState.ExitCode(), Err: waitErr}
	}
	return mediaRunResult{ExitCode: 0}
}

// mediaKillGrace is how long a cancelled engine gets to exit on its own
// before escalation (plan §5.4: 有界退出等待).
var mediaKillGrace = 8 * time.Second
