#ifndef GLOBAL_POOLING_H
#define GLOBAL_POOLING_H

#include "model.h"

template<int CH, int H, int W>
void global_avg_pool(data_t input[CH][H][W], data_t output[CH]) {
    for (int c = 0; c < CH; c++) {
        data_t sum = 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[c][h][w];
            }
        }
        output[c] = sum / (H * W);
    }
}

#endif // GLOBAL_POOLING_H 