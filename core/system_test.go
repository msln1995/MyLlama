package core

import "testing"

// ─── parseLinuxCPUModel ──────────────────────────────────────────

// TestParseLinuxCPUModel verifies extracting the "model name" line from /proc/cpuinfo
// text and stripping whitespace after the colon.
func TestParseLinuxCPUModel(t *testing.T) {
	out := "processor\t: 0\n" +
		"model name\t: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz\n" +
		"stepping\t: 5\n"
	if got := parseLinuxCPUModel(out); got != "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz" {
		t.Errorf("parseLinuxCPUModel = %q", got)
	}
	if got := parseLinuxCPUModel("no model here"); got != "" {
		t.Errorf("no model name line should return empty string, got %q", got)
	}
}

// TestCountString verifies counting lines by substring (used for Linux core count
// "processor" lines).
func TestCountString(t *testing.T) {
	out := "processor\t: 0\nprocessor\t: 1\nprocessor\t: 2\nmodel\t: x\n"
	if got := countString(out, "processor"); got != 3 {
		t.Errorf("countString = %d, want 3", got)
	}
}

// TestParseCoresDarwin verifies darwin core-count string parsing; illegal input falls
// back to 0. The empty case is the real Intel-Mac path for
// hw.perflevel0.physicalcpu (key does not exist, sysctl prints nothing) — 0
// means "split unknown", so the tuner must not cap.
func TestParseCoresDarwin(t *testing.T) {
	if got := parseCoresDarwin("8"); got != 8 {
		t.Errorf("parseCoresDarwin(8) = %d", got)
	}
	if got := parseCoresDarwin("abc"); got != 0 {
		t.Errorf("parseCoresDarwin(abc) = %d, want 0", got)
	}
	if got := parseCoresDarwin(""); got != 0 {
		t.Errorf("parseCoresDarwin(empty) = %d, want 0", got)
	}
}

// ─── parseMemInfo / parseVMStat ──────────────────────────────────

// TestParseMemInfo verifies extracting the kB value for a specified key from
// /proc/meminfo.
func TestParseMemInfo(t *testing.T) {
	out := "MemTotal:       33554432 kB\n" +
		"MemFree:        10000000 kB\n" +
		"MemAvailable:   20000000 kB\n"
	if got := parseMemInfo(out, "MemTotal"); got != 33554432 {
		t.Errorf("MemTotal = %d, want 33554432", got)
	}
	if got := parseMemInfo(out, "MemAvailable"); got != 20000000 {
		t.Errorf("MemAvailable = %d, want 20000000", got)
	}
	if got := parseMemInfo(out, "SwapTotal"); got != 0 {
		t.Errorf("non-existent key should return 0, got %d", got)
	}
}

// TestParseVMStat verifies parsing "Pages free: N" form from darwin vm_stat output.
func TestParseVMStat(t *testing.T) {
	out := "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" +
		"Pages free:                            123456.\n" +
		"Pages active:                          456789.\n"
	if got := parseVMStat(out, "free"); got != 123456 {
		t.Errorf("parseVMStat(free) = %d, want 123456", got)
	}
}

// TestDarwinAvailableBytes verifies the macOS available-memory figure counts the
// inactive queue (which the pager hands out on demand) minus the swap already in
// use, and never falls below the free-only figure it replaces. The fixtures use
// a 24 GiB M2's real proportions: 4 GiB free, 7 GiB inactive, 3.75 GiB swapped.
func TestDarwinAvailableBytes(t *testing.T) {
	const arm64PageSize = 16384
	gib := func(n float64) uint64 { return uint64(n * (1 << 30)) }
	pages := func(n float64) uint64 { return gib(n) / arm64PageSize } // GiB -> page counts
	cases := []struct {
		name     string
		free     uint64
		inactive uint64
		swap     uint64
		pageSize uint64
		want     uint64
	}{
		{"cache without swap debt counts as available", pages(4), pages(7), 0, arm64PageSize, gib(11)},
		{"measured M2 split: swap is charged against the cache", pages(4), pages(7), gib(3.75), arm64PageSize, gib(7.25)},
		{"swap deeper than the cache never drops below free-only", pages(4), pages(2), gib(6), arm64PageSize, gib(4)},
		{"empty free queue still gets the reclaimable cache", 0, pages(8), gib(1), arm64PageSize, gib(7)},
		{"4 KiB pages convert at their own size", gib(1) / 4096, 0, 0, 4096, gib(1)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := darwinAvailableBytes(c.free, c.inactive, c.swap, c.pageSize); got != c.want {
				t.Errorf("darwinAvailableBytes(%d, %d, %d, %d) = %d, want %d",
					c.free, c.inactive, c.swap, c.pageSize, got, c.want)
			}
		})
	}
}

// TestParseSwapUsageBytes verifies reading the used figure out of
// `sysctl -n vm.swapusage`, in whatever unit that line prints, with unparsable
// or absent output meaning no recorded swap debt.
func TestParseSwapUsageBytes(t *testing.T) {
	mib := func(n float64) uint64 { return uint64(n * (1 << 20)) }
	cases := []struct {
		name string
		out  string
		want uint64
	}{
		{"measured macOS line", "total = 5120.00M  used = 3885.44M  free = 1234.56M  (encrypted)", mib(3885.44)},
		{"no swap in use", "total = 1024.00M  used = 0.00M  free = 1024.00M  (encrypted)", 0},
		{"gigabyte unit", "total = 8.00G  used = 2.50G  free = 5.50G", mib(2560)},
		{"kilobyte unit", "total = 1024.00M  used = 524288.00K", mib(512)},
		{"empty output", "", 0},
		{"garbage value", "used = abcM", 0},
		{"missing used field", "total = 5120.00M  free = 5120.00M", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseSwapUsageBytes(c.out)
			// The decimal figure this tool prints is not exactly representable in
			// binary, so compare within one MiB rather than pinning a rounding.
			diff := got - c.want
			if c.want > got {
				diff = c.want - got
			}
			if diff > 1<<20 {
				t.Errorf("parseSwapUsageBytes(%q) = %d, want %d (±1 MiB)", c.out, got, c.want)
			}
		})
	}
}

// TestDefaultServerConfig verifies server config default-value fallback (accessMode local,
// host derived 127.0.0.1, port 8080).
func TestDefaultServerConfig(t *testing.T) {
	cfg := defaultServerConfig()
	if cfg.AccessMode != accessLocal || cfg.Host != "127.0.0.1" || cfg.Port != 8080 || cfg.MaxModels != 1 || cfg.CacheRAM != 8192 {
		t.Errorf("default server config does not match expected: %+v", cfg)
	}
}
