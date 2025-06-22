#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "model.h"
#include <cmath>

// Batch Normalization Layer
// Applies per-channel batch normalization to the input tensor
template<int CH, int H, int W>
void batchnorm_layer(
    data_t input[CH][H][W],   // Input feature map
    data_t gamma[CH],         // Scale parameter for each channel
    data_t beta[CH],          // Shift parameter for each channel
    data_t mean[CH],          // Mean for each channel (computed during training)
    data_t var[CH],           // Variance for each channel (computed during training)
    data_t output[CH][H][W],  // Output feature map after normalization
    float eps = 1e-5          // Small constant for numerical stability
) {
    // Iterate over channels
    for (int c = 0; c < CH; c++) {
        // Iterate over height
        for (int h = 0; h < H; h++) {
            // Iterate over width
            for (int w = 0; w < W; w++) {
                // Normalize the input value and apply scale and shift
                data_t normalized = (input[c][h][w] - mean[c]) / std::sqrt(var[c] + data_t(eps));
                output[c][h][w] = gamma[c] * normalized + beta[c];
            }
        }
    }
}

#endif // BATCHNORM_LAYER_H
