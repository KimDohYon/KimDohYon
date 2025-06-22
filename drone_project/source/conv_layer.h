#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "model.h"

// Convolution Layer
// Supports grouped convolution with configurable stride, padding, and dilation
template<int IN_CH, int OUT_CH, int IN_H, int IN_W, int OUT_H, int OUT_W, int K, int GROUP>
void conv_layer(
    data_t input[IN_CH][IN_H][IN_W],                       // Input feature map
    data_t weight[OUT_CH][IN_CH / GROUP][K][K],            // Convolution kernels
    data_t* bias,                                          // Bias for each output channel (can be nullptr)
    data_t output[OUT_CH][OUT_H][OUT_W],                   // Output feature map
    int stride,                                            // Stride for convolution
    int padding,                                           // Padding applied to input
    int group,                                             // Number of groups (for grouped convolution)
    int dilation                                           // Dilation factor for convolution
) {
    const int in_ch_per_group = IN_CH / group;             // Input channels per group
    const int out_ch_per_group = OUT_CH / group;           // Output channels per group

    // Local buffer to hold a copy of the input tensor
    data_t input_local[IN_CH][IN_H][IN_W];

    // Copy input to local buffer (used to reduce memory access latency)
    for (int c = 0; c < IN_CH; c++) {
        for (int h = 0; h < IN_H; h++) {
            for (int w = 0; w < IN_W; w++) {
                input_local[c][h][w] = input[c][h][w];
            }
        }
    }

    // Main convolution loop (supports grouped convolution)
    for (int g = 0; g < group; g++) {
#pragma HLS UNROLL
        for (int oc = 0; oc < out_ch_per_group; oc++) {
#pragma HLS UNROLL
            const int oc_global = g * out_ch_per_group + oc;  // Global output channel index

            for (int oh = 0; oh < OUT_H; oh++) {
#pragma HLS UNROLL
                for (int ow = 0; ow < OUT_W; ow++) {
#pragma HLS UNROLL
                    // Initialize sum with bias if provided
                    data_t sum = (bias != nullptr) ? bias[oc_global] : 0;

                    for (int ic = 0; ic < in_ch_per_group; ic++) {
#pragma HLS UNROLL
                        const int ic_global = g * in_ch_per_group + ic;  // Global input channel index

                        for (int kh = 0; kh < K; kh++) {
#pragma HLS UNROLL
                            for (int kw = 0; kw < K; kw++) {
#pragma HLS UNROLL
                                int ih = oh * stride - padding + kh * dilation; // Input height index
                                int iw = ow * stride - padding + kw * dilation; // Input width index

                                // Check for valid input indices
                                if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                    sum += input_local[ic_global][ih][iw] * weight[oc_global][ic][kh][kw];
                                }
                            }
                        }
                    }
                    output[oc_global][oh][ow] = sum;  // Store computed value in output
                }
            }
        }
    }
}

#endif // CONV_LAYER_H
