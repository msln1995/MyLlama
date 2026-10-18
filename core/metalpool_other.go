//go:build !(darwin && cgo)

package core

// metalWiredPoolMB has no Metal device to ask on this platform (or this is a
// tooling build without cgo): 0 means "unknown", so the tuner keeps planning
// against its RAM budget alone. See metalpool_darwin.go for the real probe.
func metalWiredPoolMB() int { return 0 }
