#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "model.h"
#include <cstddef> // size_t 정의
#ifndef __SYNTHESIS__
#include <fstream>
#include <iostream>
#include <cstdlib> // exit() 함수를 위해 추가
#endif

// 가중치 초기화 함수 선언
void initialize_weights();

// Stem 레이어 가중치
extern data_t stem_conv_weight[12][2][3][3];  // onnx::Conv_223
extern data_t stem_conv_bias[12];            // onnx::Conv_224

// Cell 0 레이어 가중치
extern data_t cell0_conv1_weight[8][12][1][1];  // onnx::Conv_226
extern data_t cell0_conv1_bias[8];             // onnx::Conv_227
extern data_t cell0_conv2_weight[8][12][1][1];  // onnx::Conv_229
extern data_t cell0_conv2_bias[8];             // onnx::Conv_230
extern data_t cell0_conv3_weight[8][8][1][1];   // onnx::Conv_232
extern data_t cell0_conv3_bias[8];             // onnx::Conv_233
extern data_t cell0_conv4_weight[8][8][1][1];   // onnx::Conv_235
extern data_t cell0_conv4_bias[8];             // onnx::Conv_236
extern data_t cell0_conv5_weight[8][8][1][1];   // onnx::Conv_238
extern data_t cell0_conv5_bias[8];             // onnx::Conv_239
extern data_t cell0_conv6_weight[8][8][1][1];   // onnx::Conv_241
extern data_t cell0_conv6_bias[8];             // onnx::Conv_242
extern data_t cell0_conv7_weight[8][8][1][1];   // onnx::Conv_244
extern data_t cell0_conv7_bias[8];             // onnx::Conv_245
extern data_t cell0_conv8_weight[8][8][1][1];   // onnx::Conv_247
extern data_t cell0_conv8_bias[8];             // onnx::Conv_248
extern data_t cell0_conv9_weight[8][8][1][1];   // onnx::Conv_250
extern data_t cell0_conv9_bias[8];             // onnx::Conv_251

// Cell 1 레이어 가중치
extern data_t cell1_conv1_weight[16][32][1][1];  // onnx::Conv_253
extern data_t cell1_conv1_bias[16];             // onnx::Conv_254
extern data_t cell1_conv2_weight[16][16][1][1];  // onnx::Conv_256
extern data_t cell1_conv2_bias[16];             // onnx::Conv_257
extern data_t cell1_conv3_weight[16][16][1][1];  // onnx::Conv_259
extern data_t cell1_conv3_bias[16];             // onnx::Conv_260
extern data_t cell1_conv4_weight[16][16][1][1];  // onnx::Conv_262
extern data_t cell1_conv4_bias[16];             // onnx::Conv_263
extern data_t cell1_conv5_weight[16][16][1][1];  // onnx::Conv_265
extern data_t cell1_conv5_bias[16];             // onnx::Conv_266
extern data_t cell1_conv6_weight[16][16][1][1];  // onnx::Conv_268
extern data_t cell1_conv6_bias[16];             // onnx::Conv_269
extern data_t cell1_conv7_weight[16][16][1][1];  // onnx::Conv_271
extern data_t cell1_conv7_bias[16];             // onnx::Conv_272
extern data_t cell1_conv8_weight[16][16][1][1];  // onnx::Conv_274
extern data_t cell1_conv8_bias[16];             // onnx::Conv_275

// Cell 1 BatchNorm 가중치
extern data_t cells_1_preprocess0_bn_weight[16];
extern data_t cells_1_preprocess0_bn_bias[16];
extern data_t cells_1_preprocess0_bn_running_mean[16];
extern data_t cells_1_preprocess0_bn_running_var[16];

// Classifier 레이어 가중치
extern data_t classifier_weight[12][64];     // classifier.weight
extern data_t classifier_bias[12];           // classifier.bias

// 추가 가중치
extern data_t cells_0_ops_by_node_0_1_op_1_weight[8][1][5][5];
extern data_t cells_0_ops_by_node_1_1_op_1_weight[8][1][3][3];
extern data_t cells_0_ops_by_node_2_0_op_1_weight[8][1][5][5];
extern data_t cells_0_ops_by_node_2_0_op_5_weight[8][1][5][5];
extern data_t cells_0_ops_by_node_2_1_op_1_weight[8][1][5][5];
extern data_t cells_0_ops_by_node_3_0_op_1_weight[8][1][5][5];
extern data_t cells_0_ops_by_node_3_0_op_5_weight[8][1][5][5];
extern data_t cells_1_preprocess0_conv1_weight[8][12][1][1];
extern data_t cells_1_preprocess0_conv2_weight[8][12][1][1];
extern data_t cells_1_ops_by_node_0_1_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_1_1_op_1_weight[16][1][3][3];
extern data_t cells_1_ops_by_node_2_0_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_2_0_op_5_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_2_1_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_3_0_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_3_0_op_5_weight[16][1][5][5];
#ifndef __SYNTHESIS__

// 1D
template<typename T, size_t N>
void load_weights_from_file(const char* filename, T (&weights)[N]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Fatal Error: Could not open weight file: " << filename << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < N; i++) {
        float temp;
        if (!(file >> temp)) {
            std::cerr << "Fatal Error: Failed to read weight value at index " << i 
                      << " from file: " << filename << std::endl;
            exit(1);
        }
        weights[i] = temp;
    }
    file.close();
    std::cout << "Successfully loaded weights from: " << filename << std::endl;
}

// 2D
template<typename T, size_t N, size_t M>
void load_weights_from_file(const char* filename, T (&weights)[N][M]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Fatal Error: Could not open weight file: " << filename << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            float temp;
            if (!(file >> temp)) {
                std::cerr << "Fatal Error: Failed to read weight value at [" << i << "][" << j 
                          << "] from file: " << filename << std::endl;
                exit(1);
            }
            weights[i][j] = temp;
        }
    }
    file.close();
    std::cout << "Successfully loaded weights from: " << filename << std::endl;
}

// 4D
template<typename T, size_t N, size_t M, size_t K, size_t L>
void load_weights_from_file(const char* filename, T (&weights)[N][M][K][L]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Fatal Error: Could not open weight file: " << filename << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            for (size_t k = 0; k < K; k++) {
                for (size_t l = 0; l < L; l++) {
                    float temp;
                    if (!(file >> temp)) {
                        std::cerr << "Fatal Error: Failed to read weight value at [" << i << "][" 
                                  << j << "][" << k << "][" << l << "] from file: " << filename << std::endl;
                        exit(1);
                    }
                    weights[i][j][k][l] = temp;
                }
            }
        }
    }
    file.close();
    std::cout << "Successfully loaded weights from: " << filename << std::endl;
}

#else

// __SYNTHESIS__ macro 활성화 시 dummy 정의로 대체 (경고 제거)
template<typename T, size_t N>
void load_weights_from_file(const char*, T (&)[N]) {}
template<typename T, size_t N, size_t M>
void load_weights_from_file(const char*, T (&)[N][M]) {}
template<typename T, size_t N, size_t M, size_t K, size_t L>
void load_weights_from_file(const char*, T (&)[N][M][K][L]) {}

#endif

#endif