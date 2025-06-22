#ifndef GEMM_LAYER_H
#define GEMM_LAYER_H

#include "model.h"

// General Matrix-Matrix Multiplication (GEMM) for 2D input/output
// Note: This form is less commonly used in practical models
template<int M, int N, int K>
void gemm_layer(
    data_t input[M][K],       // Input matrix of size M x K
    data_t weight[N][K],      // Weight matrix of size N x K (transposed)
    data_t bias[N],           // Bias vector of length N
    data_t output[M][N]       // Output matrix of size M x N
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            data_t sum = bias[n]; // Initialize with bias
            for (int k = 0; k < K; k++) {
                sum += input[m][k] * weight[n][k]; // Accumulate dot product
            }
            output[m][n] = sum; // Store result
        }
    }
}

// General Matrix-Vector Multiplication (GEMV) for 1D input/output
// This version is more commonly used in real-world models (e.g., fully connected layers)
template<int N, int K>
void gemm_layer_1d(
    data_t input[K],          // Input vector of length K
    data_t weight[N][K],      // Weight matrix of size N x K (transposed)
    data_t bias[N],           // Bias vector of length N
    data_t output[N]          // Output vector of length N
) {
    for (int n = 0; n < N; n++) {
        data_t sum = bias[n]; // Initialize with bias
        for (int k = 0; k < K; k++) {
            sum += input[k] * weight[n][k]; // Accumulate dot product
        }
        output[n] = sum; // Store result
    }
}

#endif // GEMM_LAYER_H
