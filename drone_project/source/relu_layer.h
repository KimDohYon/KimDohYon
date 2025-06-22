#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "model.h"

// ReLU Activation Layer (Template Declaration)
// Applies the ReLU function element-wise to the input tensor
template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],    // Input feature map
    data_t output[CH][H][W]    // Output feature map (ReLU-applied)
);

// ReLU Template Function Definition
template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],    // Input feature map
    data_t output[CH][H][W]    // Output feature map
) {
    // Iterate over all elements and apply ReLU: max(0, input)
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = (input[c][h][w] > data_t(0)) ? input[c][h][w] : data_t(0);
            }
        }
    }
}

#endif // RELU_LAYER_H
