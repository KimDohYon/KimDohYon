#ifndef GLOBAL_POOLING_H
#define GLOBAL_POOLING_H

#include "model.h"

// Global Average Pooling Layer
// Computes the average of all spatial elements for each channel
template<int CH, int H, int W>
void global_avg_pool(
    data_t input[CH][H][W],   // Input feature map
    data_t output[CH]         // Output vector (one value per channel)
) {
    for (int c = 0; c < CH; c++) {
        data_t sum = 0;
        // Accumulate all values within the spatial dimensions
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[c][h][w];
            }
        }
        // Compute mean for each channel
        output[c] = sum / (H * W);
    }
}

#endif // GLOBAL_POOLING_H
