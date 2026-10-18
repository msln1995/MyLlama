package core

import (
	"sync"
	"testing"
)

// resetArm64HostProbe clears the cached hardware answer and installs a stubbed
// sysctl read, so a test can drive the branches without running anything.
func resetArm64HostProbe(t *testing.T, out string) {
	t.Helper()
	arm64HostOnce = sync.Once{}
	arm64Host = false
	previous := arm64HostProbe
	arm64HostProbe = func() string { return out }
	t.Cleanup(func() {
		arm64HostProbe = previous
		arm64HostOnce = sync.Once{}
		arm64Host = false
	})
}

// TestParseSysctlFlag verifies the boolean sysctl reader: "1" wins in every
// whitespace form runCmd can leave behind, and anything else — including an
// Intel Mac's 0, a missing key, an error message and empty output — is false.
func TestParseSysctlFlag(t *testing.T) {
	for _, in := range []string{"1", "1\n", " 1 \n"} {
		if !parseSysctlFlag(in) {
			t.Errorf("parseSysctlFlag(%q) = false, want true", in)
		}
	}
	for _, in := range []string{"0", "0\n", "", "\n", "No such sysctl name", "10", "arm64"} {
		if parseSysctlFlag(in) {
			t.Errorf("parseSysctlFlag(%q) = true, want false", in)
		}
	}
}

// TestHostIsArm64 verifies the Apple Silicon answer comes from the hardware
// probe on macOS only: the sysctl is read once and cached, and no other
// platform runs anything to ask the question.
func TestHostIsArm64(t *testing.T) {
	// Intel Mac (0) and a probe that returned nothing both answer false.
	for _, out := range []string{"0\n", ""} {
		resetArm64HostProbe(t, out)
		if hostIsArm64("darwin") {
			t.Errorf("hostIsArm64(darwin) with sysctl %q = true, want false", out)
		}
	}

	resetArm64HostProbe(t, "1\n")
	if !hostIsArm64("darwin") {
		t.Fatal("hostIsArm64(darwin) with hw.optional.arm64=1 = false, want true")
	}
	// Second call must come from the cache: a probe that would now panic proves
	// runCmd is not re-run per call.
	arm64HostProbe = func() string { panic("sysctl probed twice") }
	if !hostIsArm64("darwin") {
		t.Error("cached arm64 answer lost")
	}

	// Off macOS the answer is false without touching the probe.
	arm64HostProbe = func() string { panic("probe ran off-darwin") }
	for _, goos := range []string{"windows", "linux", "android"} {
		if hostIsArm64(goos) {
			t.Errorf("hostIsArm64(%s) = true, want false", goos)
		}
	}
}

// TestEffectiveGoArch verifies the arch the rest of the app plans for: Apple
// Silicon always resolves to arm64, including the Rosetta session where
// runtime.GOARCH says amd64; without that hardware answer nothing is overridden,
// and no other platform is touched.
func TestEffectiveGoArch(t *testing.T) {
	cases := []struct {
		goos, goarch string
		hostArm64    bool
		want         string
	}{
		{"darwin", "amd64", true, "arm64"},  // Rosetta on Apple Silicon
		{"darwin", "arm64", true, "arm64"},  // normal arm64 slice
		{"darwin", "amd64", false, "amd64"}, // Intel Mac
		{"darwin", "arm64", false, "arm64"}, // probe unavailable, slice decides
		{"windows", "amd64", true, "amd64"}, // the flag never applies elsewhere
		{"linux", "amd64", true, "amd64"},
		{"android", "arm64", false, "arm64"},
	}
	for _, c := range cases {
		if got := effectiveGoArch(c.goos, c.goarch, c.hostArm64); got != c.want {
			t.Errorf("effectiveGoArch(%s, %s, %v) = %s, want %s", c.goos, c.goarch, c.hostArm64, got, c.want)
		}
	}
}

// TestDarwinHasAppleGPU verifies the GPU entry follows the machine: an arm64
// process, or an amd64 process on arm64 hardware (Rosetta), both see the
// integrated Apple GPU, and only a real Intel host gets none.
func TestDarwinHasAppleGPU(t *testing.T) {
	cases := []struct {
		goarch    string
		hostArm64 bool
		want      bool
	}{
		{"arm64", true, true},
		{"arm64", false, true},
		{"amd64", true, true},
		{"amd64", false, false},
	}
	for _, c := range cases {
		if got := darwinHasAppleGPU(c.goarch, c.hostArm64); got != c.want {
			t.Errorf("darwinHasAppleGPU(%s, %v) = %v, want %v", c.goarch, c.hostArm64, got, c.want)
		}
	}
}

// TestHostAssetArchMatchesHost runs the production probe path (hostIsArm64 →
// runCmd → sysctl) on the machine executing the test and checks it lands on an
// arch tag the release picker can actually match.
func TestHostAssetArchMatchesHost(t *testing.T) {
	got := hostAssetArch()
	if archKeyOf(got) == "" {
		t.Errorf("hostAssetArch() = %q, which the release picker cannot map to an asset tag", got)
	}
}
