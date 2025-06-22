#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "model.h"
#include <cfloat>

// Max Pooling Layer
// Applies a KxK max pooling operation with stride and padding
template<int CH, int K, int H, int W, int STRIDE>
void maxpool_layer(
    data_t input[CH][H][W],  // Input feature map
    data_t output[CH][(H + 2 - K) / STRIDE + 1][(W + 2 - K) / STRIDE + 1], // Output feature map
    int stride,              // Stride value for pooling
    int padding              // Padding size (applied symmetrically)
) {
    const int OH = (H + 2 * padding - K) / stride + 1;  // Output height
    const int OW = (W + 2 * padding - K) / stride + 1;  // Output width
    const int PH = H + 2 * padding;                    // Padded input height
    const int PW = W + 2 * padding;                    // Padded input width

    data_t padded[CH][PH][PW];  // Zero-padded input buffer

    // Pad the input with -FLT_MAX for max pooling (acts as neutral element)
    for (int c = 0; c < CH; c++) {
        for (int i = 0; i < PH; i++) {
            for (int j = 0; j < PW; j++) {
                if (i >= padding && i < H + padding && j >= padding && j < W + padding)
                    padded[c][i][j] = input[c][i - padding][j - padding]; // Copy original input
                else
                    padded[c][i][j] = -FLT_MAX; // Pad with minimum value for max pooling
            }
        }
    }

    // Perform max pooling over each channel
    for (int c = 0; c < CH; c++) {
#pragma HLS LOOP_FLATTEN off
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
#pragma HLS PIPELINE II=4
                data_t max_val = -FLT_MAX;
                // Apply KxK kernel
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        max_val = (padded[c][ih][iw] > max_val) ? padded[c][ih][iw] : max_val;
                    }
                }
                output[c][oh][ow] = max_val; // Store max value
            }
        }
    }
}

#endif // MAXPOOL_LAYER_H
