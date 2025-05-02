#pragma once
#include "shape_config.h"
#include <algorithm>

// Conv2D
inline void conv2d_module(const float input_norm[INPUT_H][INPUT_W],
                          float conv_out[CONV_OUT_CH][CONV_OUT_H][CONV_OUT_W],
                          const float conv_weights[CONV_OUT_CH][1][KERNEL_H][KERNEL_W],
                          const float conv_bias[CONV_OUT_CH]) {
    for (int oc = 0; oc < CONV_OUT_CH; ++oc) {
        for (int oh = 0; oh < CONV_OUT_H; ++oh) {
            float sum = conv_bias[oc];
#pragma HLS PIPELINE II=1
            for (int kh = 0; kh < KERNEL_H; ++kh) {
                for (int kw = 0; kw < KERNEL_W; ++kw) {
                    float val = input_norm[oh + kh][kw];
                    float weight = conv_weights[oc][0][kh][kw];
                    sum += val * weight;
                }
            }
            conv_out[oc][oh][0] = sum;
        }
    }
}

// FC
template<int IN, int OUT>
inline void fc_module(const float in[IN], float out[OUT],
                      const float weights[OUT][IN],
                      const float bias[OUT]) {
OUT_LOOP:
    for (int o = 0; o < OUT; ++o) {
        float sum_local = bias[o];  // 핵심: sum → sum_local

    IN_LOOP:
        for (int i = 0; i < IN; ++i) {
            sum_local += in[i] * weights[o][i];
        }
        out[o] = sum_local;
    }
}




inline void fc1_module(const float in[FC1_IN], float out[FC1_OUT],
                       const float weights[FC1_OUT][FC1_IN], const float bias[FC1_OUT]) {
    fc_module<FC1_IN, FC1_OUT>(in, out, weights, bias);
}
inline void fc2_module(const float in[FC2_IN], float out[FC2_OUT],
                       const float weights[FC2_OUT][FC2_IN], const float bias[FC2_OUT]) {
    fc_module<FC2_IN, FC2_OUT>(in, out, weights, bias);
}
inline void fc3_module(const float in[FC3_IN], float out[FC3_OUT],
                       const float weights[FC3_OUT][FC3_IN], const float bias[FC3_OUT]) {
    fc_module<FC3_IN, FC3_OUT>(in, out, weights, bias);
}
inline void fc4_module(const float in[FC4_IN], float out[FC4_OUT],
                       const float weights[FC4_OUT][FC4_IN], const float bias[FC4_OUT]) {
    fc_module<FC4_IN, FC4_OUT>(in, out, weights, bias);
}

// ReLU
template<int CH, int H, int W>
inline void relu_module(const float in[CH][H][W], float out[CH][H][W]) {
    for (int c = 0; c < CH; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                out[c][i][j] = std::max(0.0f, in[c][i][j]);
}
template<int LEN>
inline void relu_fc(float vec[LEN]) {
    for (int i = 0; i < LEN; ++i)
        if (vec[i] < 0) vec[i] = 0;
}

// MaxPooling
inline void maxpool_module(const float in[CONV_OUT_CH][CONV_OUT_H][CONV_OUT_W],
                           float out[CONV_OUT_CH][POOL_OUT_H][POOL_OUT_W]) {
    for (int c = 0; c < CONV_OUT_CH; ++c)
        for (int ph = 0; ph < POOL_OUT_H; ++ph) {
            float maxval = -1e9f;
            for (int kh = 0; kh < POOL_KH; ++kh) {
                int ih = ph * POOL_KH + kh;
                maxval = std::max(maxval, in[c][ih][0]);
            }
            out[c][ph][0] = maxval;
        }
}

// Flatten
inline void flatten_module(const float in[CONV_OUT_CH][POOL_OUT_H][POOL_OUT_W],
                           float out[FLAT_SIZE]) {
    int idx = 0;
    for (int c = 0; c < CONV_OUT_CH; ++c)
        for (int h = 0; h < POOL_OUT_H; ++h)
            for (int w = 0; w < POOL_OUT_W; ++w)
                out[idx++] = in[c][h][w];
}