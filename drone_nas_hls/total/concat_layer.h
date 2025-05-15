#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H
#include "model.h"
// 4개 입력 concat (직접 배열 인자)
// 입력: inputs[4][CH][H][W], 출력: output[4*CH][H][W]
template<int CH, int H, int W>
void concat_layer(
    data_t in0[CH][H][W],
    data_t in1[CH][H][W],
    data_t in2[CH][H][W],
    data_t in3[CH][H][W],
    data_t output[4*CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = in0[c][h][w];
                output[CH + c][h][w] = in1[c][h][w];
                output[2*CH + c][h][w] = in2[c][h][w];
                output[3*CH + c][h][w] = in3[c][h][w];
            }
        }
    }
}
// 2개 입력 concat (직접 배열 인자)
// 입력: in1, in2, 출력: output[2*CH][H][W]
template<int CH, int H, int W>
void concat2_layer(
    data_t in0[CH][H][W],
    data_t in1[CH][H][W],
    data_t output[2*CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = in0[c][h][w];
                output[CH + c][h][w] = in1[c][h][w];
            }
        }
    }
}
#endif // CONCAT_LAYER_H 