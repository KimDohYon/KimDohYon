// conv_wrapper.h
#pragma once
#include "conv_layer.h"
#include "model.h"

// Depthwise 5x5 convolution wrapper
template<int CH, int IN_H, int IN_W, int OUT_H, int OUT_W>
void depthwise_conv5x5(
    data_t input[CH][IN_H][IN_W],
    data_t weight[CH][1][5][5],
    data_t* bias,
    data_t output[CH][OUT_H][OUT_W],
    int stride = 1, int padding = 2, int dilation = 1
) {
#pragma HLS inline off
    conv_layer<CH, CH, IN_H, IN_W, OUT_H, OUT_W, 5, CH>(
        input, weight, bias, output, stride, padding, CH, dilation);
}

// Depthwise 3x3 convolution wrapper
template<int CH, int IN_H, int IN_W, int OUT_H, int OUT_W>
void depthwise_conv3x3(
    data_t input[CH][IN_H][IN_W],
    data_t weight[CH][1][3][3],
    data_t* bias,
    data_t output[CH][OUT_H][OUT_W],
    int stride = 1, int padding = 1, int dilation = 1
) {
#pragma HLS inline off
    conv_layer<CH, CH, IN_H, IN_W, OUT_H, OUT_W, 3, CH>(
        input, weight, bias, output, stride, padding, CH, dilation);
}

// Pointwise 1x1 convolution wrapper
template<int IN_CH, int OUT_CH, int IN_H, int IN_W>
void pointwise_conv1x1(
    data_t input[IN_CH][IN_H][IN_W],
    data_t weight[OUT_CH][IN_CH][1][1],
    data_t* bias,
    data_t output[OUT_CH][IN_H][IN_W]
) {
#pragma HLS inline off
    conv_layer<IN_CH, OUT_CH, IN_H, IN_W, IN_H, IN_W, 1, 1>(
        input, weight, bias, output, 1, 0, 1, 1);
}