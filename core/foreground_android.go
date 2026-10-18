//go:build android

package core

import (
	"github.com/wailsapp/wails/v3/pkg/application"
)

// Android implementation: forward the declared WailsForegroundService
// (dataSync, ongoing notification) to the Java bridge. Title and body travel
// as the JSON object the bridge parses in WailsBridge.startForegroundService.
func keepAwakeOn(title, text string) {
	application.Android.StartForegroundService(foregroundPayload(title, text))
}

func keepAwakeOff() {
	application.Android.StopForegroundService()
}
