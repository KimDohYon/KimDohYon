#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "model.h"

template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],
    data_t output[CH][H][W]
);

// ReLU 템플릿 함수 구현

template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],
    data_t output[CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = (input[c][h][w] > data_t(0)) ? input[c][h][w] : data_t(0);
            }
        }
    }
}

#endif // RELU_LAYER_H 