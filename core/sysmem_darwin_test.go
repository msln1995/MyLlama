//go:build darwin

package core

import (
	"strconv"
	"strings"
	"testing"
)

// TestDarwinMemoryBytesOnHost checks the macOS probe's wiring on a real Mac: the
// available figure must sit inside physical RAM and never below the free-only
// reading it replaced, since a figure above total would let a plan budget RAM the
// machine does not have.
func TestDarwinMemoryBytesOnHost(t *testing.T) {
	total, available, ok := darwinMemoryBytes()
	if !ok || total == 0 {
		t.Skip("no vm_stat answer on this host")
	}
	pageSize := uint64(16384)
	if ps, err := strconv.ParseUint(strings.TrimSpace(runCmd("sysctl", "-n", "hw.pagesize")), 10, 64); err == nil && ps > 0 {
		pageSize = ps
	}
	freeOnly := parseVMStat(runCmd("vm_stat"), "free") * pageSize
	if available > total {
		t.Errorf("available = %d bytes, above physical RAM %d", available, total)
	}
	if available < freeOnly {
		t.Errorf("available = %d, below the free-only figure %d it replaces", available, freeOnly)
	}
	gib := func(v uint64) float64 { return float64(v) / (1 << 30) }
	t.Logf("darwin probe: total=%.1f GiB available=%.1f GiB (free-only %.1f GiB)", gib(total), gib(available), gib(freeOnly))
}
