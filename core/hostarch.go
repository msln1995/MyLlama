package core

import (
	"runtime"
	"strings"
	"sync"
)

// ─── Host architecture vs. the slice this process was launched as ──────
//
// runtime.GOARCH answers "which slice did the loader pick", not "what machine
// is this", and on Apple Silicon the two come apart whenever the app runs
// translated (an x86_64-only bundle, or the Finder's "Open using Rosetta").
// Every consumer of GOARCH then treats the Mac as an Intel box: the release
// picker asks for the macos-x64 tarball, which upstream builds without the
// Metal backend (measured against llama-b10537-bin-macos-x64.tar.gz — 59
// entries, libggml-base/-blas/-cpu only, while the arm64 tarball carries
// libggml-metal.0.<ver>.dylib), and the GPU probe returns no device, so the
// auto-tuner plans a CPU run on a machine that is one download away from Metal.
//
// Translation does not follow child processes: a translated parent that execs
// an arm64 binary gets it running natively. Measured on the M2 this was written
// against (macOS 27.2, Rosetta 2 installed): the amd64 slice reports
// sysctl.proc_translated=1 and the arm64 child it spawned reported
// proc_translated=0 / goarch=arm64. Choosing the arm64 build from a Rosetta
// session therefore costs nothing and keeps the GPU path.
//
// hw.optional.arm64 is the hardware fact and it reads 1 from both slices, so it
// decides.

// arm64HostProbeFn reads the sysctl; the seam lets tests drive the branches
// without an exec (same style as the other platform probes).
var (
	arm64HostOnce  sync.Once
	arm64Host      bool
	arm64HostProbe = arm64HostProbeFn
)

func arm64HostProbeFn() string {
	return runCmd("sysctl", "-n", "hw.optional.arm64")
}

// hostIsArm64 reports whether the machine is Apple Silicon regardless of the
// arch this process runs as. Only macOS is asked — every other platform
// answers false without running anything — and the probe happens once per
// process like the other hardware facts.
func hostIsArm64(goos string) bool {
	if goos != "darwin" {
		return false
	}
	arm64HostOnce.Do(func() {
		arm64Host = parseSysctlFlag(arm64HostProbe())
	})
	return arm64Host
}

// parseSysctlFlag reads a boolean sysctl answer ("1" / "0" plus the newline
// runCmd leaves). Anything that is not exactly 1 is false, which keeps an Intel
// Mac, an unknown key and a failed probe in the same bucket.
func parseSysctlFlag(out string) bool {
	return strings.TrimSpace(out) == "1"
}

// effectiveGoArch is the arch to plan for: Apple Silicon hardware always
// answers arm64, other platforms keep the process arch.
func effectiveGoArch(goos, goarch string, hostArm64 bool) string {
	if goos == "darwin" && hostArm64 {
		return "arm64"
	}
	return goarch
}

// hostAssetArch is the arch the llama.cpp release picker matches asset names
// against on this machine.
func hostAssetArch() string {
	return effectiveGoArch(runtime.GOOS, runtime.GOARCH, hostIsArm64(runtime.GOOS))
}

// darwinHasAppleGPU reports whether the integrated Apple GPU entry exists. The
// Metal-capable llama.cpp release is the arm64 build and an arm64 machine is
// what can run it, so the slice this process happens to be in is not the
// question.
func darwinHasAppleGPU(goarch string, hostArm64 bool) bool {
	return goarch == "arm64" || hostArm64
}
