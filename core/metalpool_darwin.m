// Metal's wired-memory ceiling, read from the device itself.

#import <Metal/Metal.h>

// myllamaMetalWiredPoolBytes is the largest working set Metal will let a
// process keep resident (MTLDevice's recommendedMaxWorkingSetSize). Returns 0
// when no Metal device exists.
long long myllamaMetalWiredPoolBytes(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return 0;
    }
    long long bytes = (long long)[device recommendedMaxWorkingSetSize];
    [device release];
    return bytes;
}
