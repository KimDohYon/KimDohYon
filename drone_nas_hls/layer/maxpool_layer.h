#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "model.h"
#include <cfloat>

template<int CH, int K, int H, int W, int STRIDE>
void maxpool_layer(
    data_t input[CH][H][W],
    data_t output[CH][(H + 2 - K)/STRIDE + 1][(W + 2 - K)/STRIDE + 1], // 패딩 고려
    int stride,
    int padding
) {
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;
    const int PH = H + 2 * padding;
    const int PW = W + 2 * padding;

    data_t padded[CH][PH][PW];

    // Initialize padded input with 0 (or MIN_FLOAT if needed)
    for (int c = 0; c < CH; c++) {
        for (int i = 0; i < PH; i++) {
            for (int j = 0; j < PW; j++) {
                if (i >= padding && i < H + padding && j >= padding && j < W + padding)
                    padded[c][i][j] = input[c][i - padding][j - padding];
                else
                    padded[c][i][j] = -FLT_MAX;  // for max pooling
            }
        }
    }

    for (int c = 0; c < CH; c++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                data_t max_val = -FLT_MAX;
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        max_val = (padded[c][ih][iw] > max_val) ? padded[c][ih][iw] : max_val;
                    }
                }
                output[c][oh][ow] = max_val;
            }
        }
    }
}


#endif // MAXPOOL_LAYER_H 