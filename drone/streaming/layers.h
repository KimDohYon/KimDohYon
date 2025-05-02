#pragma once
#include "shape_config.h"
#include <algorithm>
#include <hls_stream.h>

// =======================
// STREAM-BASED CONV-RELU-POOL
// =======================

// Conv2D → Stream Output
inline void conv2d_stream(const float input_norm[INPUT_H][INPUT_W],
                          hls::stream<float> &conv_relu_stream,
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
            conv_relu_stream.write(sum);
        }
    }
}

// ReLU → Stream Output
inline void relu_stream(hls::stream<float> &in, hls::stream<float> &out, int total_len) {
    for (int i = 0; i < total_len; ++i) {
#pragma HLS PIPELINE II=1
        float val = in.read();
        out.write(val > 0 ? val : 0);
    }
}

// Maxpool → Final Output Buffer
inline void maxpool_stream(hls::stream<float> &in, float out[CONV_OUT_CH][POOL_OUT_H][POOL_OUT_W]) {
    for (int c = 0; c < CONV_OUT_CH; ++c) {
        for (int ph = 0; ph < POOL_OUT_H; ++ph) {
            float maxval = -1e9f;
            for (int kh = 0; kh < POOL_KH; ++kh) {
#pragma HLS PIPELINE II=1
                float v = in.read();
                maxval = std::max(maxval, v);
            }
            out[c][ph][0] = maxval;
        }
    }
}

// =======================
// REMAINING MODULES SAME
// =======================

template<int IN, int OUT>
inline void fc_module(const float in[IN], float out[OUT],
                      const float weights[OUT][IN], const float bias[OUT]) {
    for (int o = 0; o < OUT; ++o) {
        float sum_local = bias[o];
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

template<int LEN>
inline void relu_fc(float vec[LEN]) {
    for (int i = 0; i < LEN; ++i)
        if (vec[i] < 0) vec[i] = 0;
}

inline void flatten_module(const float in[CONV_OUT_CH][POOL_OUT_H][POOL_OUT_W],
                           float out[FLAT_SIZE]) {
    int idx = 0;
    for (int c = 0; c < CONV_OUT_CH; ++c)
        for (int h = 0; h < POOL_OUT_H; ++h)
            for (int w = 0; w < POOL_OUT_W; ++w)
                out[idx++] = in[c][h][w];
}