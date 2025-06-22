#ifndef ADD_LAYER_H
#define ADD_LAYER_H

#include "model.h"

// Element-wise addition layer
// Adds two input feature maps element-wise and stores the result in the output
template<int CH, int H, int W>
void add_layer(
    data_t input1[CH][H][W],  // First input feature map
    data_t input2[CH][H][W],  // Second input feature map
    data_t output[CH][H][W]   // Output feature map (result of addition)
) {
    // Iterate over channels
    for (int c = 0; c < CH; c++) {
        // Iterate over height
        for (int h = 0; h < H; h++) {
            // Iterate over width
            for (int w = 0; w < W; w++) {
                // Perform element-wise addition
                output[c][h][w] = input1[c][h][w] + input2[c][h][w];
            }
        }
    }
}

#endif // ADD_LAYER_H
