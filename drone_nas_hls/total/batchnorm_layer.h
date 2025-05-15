#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "model.h"
#include <cmath>

template<int CH, int H, int W>
void batchnorm_layer(
    data_t input[CH][H][W],
    data_t gamma[CH],
    data_t beta[CH],
    data_t mean[CH],
    data_t var[CH],
    data_t output[CH][H][W],
    float eps = 1e-5
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                data_t normalized = (input[c][h][w] - mean[c]) / std::sqrt(var[c] + data_t(eps));
                output[c][h][w] = gamma[c] * normalized + beta[c];
            }
        }
    }
}

#endif // BATCHNORM_LAYER_H 