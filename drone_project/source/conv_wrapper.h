// conv_wrapper.h
#pragma once

#include "conv_layer.h"
#include "model.h"

// Depthwise 5x5 Convolution Wrapper
// Applies a 5x5 depthwise convolution (one filter per input channel)
template<int CH, int IN_H, int IN_W, int OUT_H, int OUT_W>
void depthwise_conv5x5(
    data_t input[CH][IN_H][IN_W],             // Input feature map
    data_t weight[CH][1][5][5],               // Depthwise convolution weights (one filter per channel)
    data_t* bias,                             // Optional bias per channel
    data_t output[CH][OUT_H][OUT_W],          // Output feature map
    int stride = 1,                           // Stride value (default: 1)
    int padding = 2,                          // Padding size (default: 2 to preserve spatial dimensions)
    int dilation = 1                          // Dilation rate (default: 1)
) {
    // Each input channel is convolved independently (group = CH)
    conv_layer<CH, CH, IN_H, IN_W, OUT_H, OUT_W, 5, CH>(
        input, weight, bias, output, stride, padding, CH, dilation);
}

// Depthwise 3x3 Convolution Wrapper
// Applies a 3x3 depthwise convolution (one filter per input channel)
template<int CH, int IN_H, int IN_W, int OUT_H, int OUT_W>
void depthwise_conv3x3(
    data_t input[CH][IN_H][IN_W],             // Input feature map
    data_t weight[CH][1][3][3],               // Depthwise convolution weights
    data_t* bias,                             // Optional bias per channel
    data_t output[CH][OUT_H][OUT_W],          // Output feature map
    int stride = 1,                           // Stride value (default: 1)
    int padding = 1,                          // Padding size (default: 1 to preserve spatial dimensions)
    int dilation = 1                          // Dilation rate (default: 1)
) {
    // Each input channel is convolved independently (group = CH)
    conv_layer<CH, CH, IN_H, IN_W, OUT_H, OUT_W, 3, CH>(
        input, weight, bias, output, stride, padding, CH, dilation);
}

// Pointwise 1x1 Convolution Wrapper
// Applies a 1x1 convolution to perform channel mixing
template<int IN_CH, int OUT_CH, int IN_H, int IN_W>
void pointwise_conv1x1(
    data_t input[IN_CH][IN_H][IN_W],          // Input feature map
    data_t weight[OUT_CH][IN_CH][1][1],       // Pointwise convolution weights (1x1)
    data_t* bias,                             // Optional bias per output channel
    data_t output[OUT_CH][IN_H][IN_W]         // Output feature map
) {
    // Group = 1: standard convolution, no channel separation
    conv_layer<IN_CH, OUT_CH, IN_H, IN_W, IN_H, IN_W, 1, 1>(
        input, weight, bias, output, 1, 0, 1, 1);
}
