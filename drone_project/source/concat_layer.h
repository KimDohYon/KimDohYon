#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "model.h"

// Concatenation Layer (4-way channel-wise)
// Concatenates 4 input feature maps along the channel axis
template<int CH, int H, int W>
void concat_layer(
    data_t in0[CH][H][W],                  // First input feature map
    data_t in1[CH][H][W],                  // Second input feature map
    data_t in2[CH][H][W],                  // Third input feature map
    data_t in3[CH][H][W],                  // Fourth input feature map
    data_t output[4 * CH][H][W]            // Output feature map with 4*CH channels
) {
    for (int c = 0; c < CH; c++) {
#pragma HLS PIPELINE off
        for (int h = 0; h < H; h++) {
#pragma HLS PIPELINE off
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE off
                output[c][h][w]         = in0[c][h][w];
                output[CH + c][h][w]    = in1[c][h][w];
                output[2 * CH + c][h][w] = in2[c][h][w];
                output[3 * CH + c][h][w] = in3[c][h][w];
            }
        }
    }
}

// Concatenation Layer (2-way channel-wise)
// Concatenates 2 input feature maps along the channel axis
template<int CH, int H, int W>
void concat2_layer(
    data_t in0[CH][H][W],                  // First input feature map
    data_t in1[CH][H][W],                  // Second input feature map
    data_t output[2 * CH][H][W]            // Output feature map with 2*CH channels
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w]       = in0[c][h][w];
                output[CH + c][h][w]  = in1[c][h][w];
            }
        }
    }
}

#endif // CONCAT_LAYER_H
