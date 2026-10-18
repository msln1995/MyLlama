//go:build darwin && cgo

package core

import "testing"

// TestMetalWiredPoolMB checks the probe's wiring on a real Mac: the figure is
// the device's working set, so it must come back in MiB (a bytes/MiB mix-up
// would put it above physical RAM) and never block a plan when the machine has
// no Metal device (0 keeps the tuner on its RAM budget).
func TestMetalWiredPoolMB(t *testing.T) {
	pool := metalWiredPoolMB()
	if pool == 0 {
		t.Skip("no Metal device on this host")
	}
	totalMB := getTotalMemoryGB() * 1024
	if float64(pool) > totalMB*1.05 {
		t.Errorf("metal pool = %d MiB, above physical RAM %.0f MiB — the unit conversion is wrong", pool, totalMB)
	}
	t.Logf("metal working-set pool = %d MiB of %.0f MiB RAM", pool, totalMB)
}
