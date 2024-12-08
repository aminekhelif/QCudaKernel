#include <cuda_runtime.h>

// A simple auto-tuner that chooses a tile size.
// Currently just sets a fixed value, but can be extended.

static int selectedTileSize = 16;

extern "C" void autoTuneKernelParams() {
    // In a real scenario, microbenchmarking logic would go here.
    selectedTileSize = 16;
}

extern "C" int getSelectedTileSize() {
    return selectedTileSize;
}