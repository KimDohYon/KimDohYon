#ifndef GEMM_LAYER_H
#define GEMM_LAYER_H

#include "model.h"

// 2D 입력/출력용 GEMM (일반적으로 잘 사용하지 않음)
template<int M, int N, int K>
void gemm_layer(
    data_t input[M][K],
    data_t weight[N][K],
    data_t bias[N],
    data_t output[M][N]
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            data_t sum = bias[n];
            for (int k = 0; k < K; k++) {
                sum += input[m][k] * weight[n][k];
            }
            output[m][n] = sum;
        }
    }
}

// 1D 입력/출력용 GEMM (실제 모델에서 주로 사용)
template<int N, int K>
void gemm_layer_1d(
    data_t input[K],
    data_t weight[N][K],
    data_t bias[N],
    data_t output[N]
) {
    for (int n = 0; n < N; n++) {
        data_t sum = bias[n];
        for (int k = 0; k < K; k++) {
            sum += input[k] * weight[n][k];
        }
        output[n] = sum;
    }
}

#endif // GEMM_LAYER_H 