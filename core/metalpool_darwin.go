//go:build darwin && cgo

package core

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
extern long long myllamaMetalWiredPoolBytes(void);
*/
import "C"

// metalWiredPoolMB reads the Metal working-set ceiling off the default device
// (the ObjC shim in metalpool_darwin.m). This is the same figure llama.cpp
// prints for its MTL0 device: on a 24 GiB M2 the call returns 18186 MiB, byte
// for byte what the server's own device listing reports. Returns 0 when no
// Metal device answers, which leaves a plan on its RAM budget alone.
func metalWiredPoolMB() int {
	bytes := int64(C.myllamaMetalWiredPoolBytes())
	if bytes <= 0 {
		return 0
	}
	return int(bytes / (1 << 20))
}
