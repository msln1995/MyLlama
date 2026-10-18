package core

import (
	"runtime"
	"slices"
	"testing"
)

// ─── parseAndroidCPUModel ────────────────────────────────────────

// TestParseAndroidCPUModel verifies the SoC identity fallback for /proc/cpuinfo
// text without an x86 "model name" line: the "Hardware" line wins, then the
// ro.soc.* build.prop properties, then "" (the caller keeps its generic
// "(unknown model)" fallback).
func TestParseAndroidCPUModel(t *testing.T) {
	// Snapdragon-style: Android kernels append a "Hardware" line to cpuinfo.
	cpuinfo := "Processor\t: AArch64 Processor rev 1 (aarch64)\n" +
		"processor\t: 0\n" +
		"BogoMIPS\t: 38.40\n" +
		"Hardware\t: Qualcomm Technologies, Inc SM8550\n"
	if got := parseAndroidCPUModel(cpuinfo, ""); got != "Qualcomm Technologies, Inc SM8550" {
		t.Errorf("Hardware line: parseAndroidCPUModel = %q", got)
	}

	// Dimensity-style: no Hardware line; build.prop carries the SoC identity.
	buildProp := "# begin build properties\n" +
		"ro.product.model=phone\n" +
		"ro.soc.manufacturer=Mediatek\n" +
		"ro.soc.model=MT6985\n"
	if got := parseAndroidCPUModel("processor\t: 0\n", buildProp); got != "Mediatek MT6985" {
		t.Errorf("build.prop: parseAndroidCPUModel = %q, want %q", got, "Mediatek MT6985")
	}

	// Model-only or manufacturer-only build.prop entries are used alone.
	if got := parseAndroidCPUModel("", "ro.soc.model=SM8550\n"); got != "SM8550" {
		t.Errorf("model only: parseAndroidCPUModel = %q, want %q", got, "SM8550")
	}
	if got := parseAndroidCPUModel("", "ro.soc.manufacturer=Qualcomm\n"); got != "Qualcomm" {
		t.Errorf("manufacturer only: parseAndroidCPUModel = %q, want %q", got, "Qualcomm")
	}

	// No signal anywhere: empty string keeps the generic fallback in charge.
	if got := parseAndroidCPUModel("", ""); got != "" {
		t.Errorf("empty inputs: parseAndroidCPUModel = %q, want empty", got)
	}

	// An empty Hardware value falls through to build.prop.
	if got := parseAndroidCPUModel("Hardware\t:\n", "ro.soc.manufacturer=Qualcomm\nro.soc.model=SM8550\n"); got != "Qualcomm SM8550" {
		t.Errorf("empty Hardware value: parseAndroidCPUModel = %q, want %q", got, "Qualcomm SM8550")
	}
}

// ─── countPerformanceCPUs ────────────────────────────────────────

// TestCountPerformanceCPUs verifies the big.LITTLE performance-core count:
// CPUs within 80% of the highest max frequency count as performance cores;
// unavailable or degenerate data returns 0.
func TestCountPerformanceCPUs(t *testing.T) {
	// Snapdragon-style 4+4 split (kHz values): only the big cluster qualifies.
	bigLittle := []int{1804800, 1804800, 1804800, 1804800, 2841600, 2841600, 2841600, 2841600}
	if got := countPerformanceCPUs(bigLittle); got != 4 {
		t.Errorf("4+4 split: countPerformanceCPUs = %d, want 4", got)
	}

	// 1+3+4 topology: prime + the three big cores count, little cluster not.
	triCluster := []int{1804800, 1804800, 1804800, 1804800, 2496000, 2496000, 2496000, 2841600}
	if got := countPerformanceCPUs(triCluster); got != 4 {
		t.Errorf("1+3+4 split: countPerformanceCPUs = %d, want 4", got)
	}

	// Boundary: exactly 80% of max still counts, one step below does not.
	if got := countPerformanceCPUs([]int{2000, 1600}); got != 2 {
		t.Errorf("boundary at 80%%: countPerformanceCPUs = %d, want 2", got)
	}
	if got := countPerformanceCPUs([]int{2000, 1599}); got != 1 {
		t.Errorf("below 80%%: countPerformanceCPUs = %d, want 1", got)
	}

	// Uniform topology: every core counts (no split → no capping downstream).
	if got := countPerformanceCPUs([]int{2000000, 2000000, 2000000}); got != 3 {
		t.Errorf("uniform: countPerformanceCPUs = %d, want 3", got)
	}

	// Degenerate inputs: no data, empty entries, or no positive frequency.
	if got := countPerformanceCPUs(nil); got != 0 {
		t.Errorf("nil: countPerformanceCPUs = %d, want 0", got)
	}
	if got := countPerformanceCPUs([]int{}); got != 0 {
		t.Errorf("empty: countPerformanceCPUs = %d, want 0", got)
	}
	if got := countPerformanceCPUs([]int{0, 0, 0}); got != 0 {
		t.Errorf("all zeros: countPerformanceCPUs = %d, want 0", got)
	}
}

// ─── CPU presence list ───────────────────────────────────────────

func pinProbeCPUPresent(t *testing.T, out string) {
	t.Helper()
	orig := probeCPUPresent
	probeCPUPresent = func() string { return out }
	t.Cleanup(func() { probeCPUPresent = orig })
}

// TestParseCPUPresentList covers the kernel CPU list format (ranges, singles,
// mixed and unordered lists, duplicate collapse) and the guards: malformed
// tokens, reversed ranges and implausibly wide ranges are skipped, leaving nil
// when nothing usable was parsed.
func TestParseCPUPresentList(t *testing.T) {
	tests := []struct {
		in   string
		want []int
	}{
		{"0-7\n", []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"0-3,4-7", []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"4-7,0-3", []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"0,2,4", []int{0, 2, 4}},
		{"0-2,1-3", []int{0, 1, 2, 3}},
		{"", nil},
		{"garbage", nil},
		{"-1", nil},
		{"3-1", nil},
		{"0-2147483647", nil},
	}
	for _, tt := range tests {
		if got := parseCPUPresentList(tt.in); !slices.Equal(got, tt.want) {
			t.Errorf("parseCPUPresentList(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

// TestCPUPresentIDsUsesHardwareTruth is the regression for the affinity-skewed
// topology sample: an Android app confined to a little-core cpuset reports
// fewer CPUs through runtime.NumCPU than the SoC has, which used to shrink both
// the frequency sample and the reported core count. The presence list decides.
func TestCPUPresentIDsUsesHardwareTruth(t *testing.T) {
	pinProbeCPUPresent(t, "0-7")
	ids := cpuPresentIDs()
	if !slices.Equal(ids, []int{0, 1, 2, 3, 4, 5, 6, 7}) {
		t.Errorf("cpuPresentIDs = %v, want the 8 CPUs the hardware presents", ids)
	}

	// Unreadable sysfs keeps the historical 0..NumCPU-1 range.
	pinProbeCPUPresent(t, "")
	ids = cpuPresentIDs()
	if len(ids) != runtime.NumCPU() {
		t.Fatalf("cpuPresentIDs fallback = %v, want %d entries", ids, runtime.NumCPU())
	}
	for i, id := range ids {
		if id != i {
			t.Errorf("fallback ids[%d] = %d, want %d", i, id, i)
		}
	}
}

// TestPresentCoreCount verifies the presence list can only raise the sampled
// processor count, never lower it, and that an unreadable list falls back to
// the affinity-based count.
func TestPresentCoreCount(t *testing.T) {
	pinProbeCPUPresent(t, "0-7")
	if got := presentCoreCount(4); got != 8 {
		t.Errorf("presentCoreCount(4) = %d, want 8 (a cpuset-confined sample)", got)
	}
	if got := presentCoreCount(12); got != 12 {
		t.Errorf("presentCoreCount(12) = %d, want 12 (never lowered)", got)
	}

	pinProbeCPUPresent(t, "")
	if got := presentCoreCount(0); got != runtime.NumCPU() {
		t.Errorf("presentCoreCount(0) fallback = %d, want NumCPU %d", got, runtime.NumCPU())
	}
}
