// cell_layer.h
#pragma once
#include "model.h"
#include "relu_layer.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "maxpool_layer.h"
#include "conv_wrapper.h"

template<int CH, int IN_H, int IN_W>
void cell_layer(
    // 입력 텐서들 (preprocess 출력 두 개) 
    data_t inputA[CH][IN_H][IN_W],
    data_t inputB[CH][IN_H][IN_W],
    // 출력 텐서 (4*CH 채널, 크기는 절반) 
    data_t output[4*CH][IN_H/2][IN_W/2],
    // 각 Conv 연산의 가중치와 바이어스 파라미터들
    // Node 0.1 (Conv 0, Conv 1)
    data_t conv_weight0[CH][1][5][5], data_t* conv_bias0,
    data_t conv_weight1[CH][CH][1][1], data_t* conv_bias1,
    // Node 1.1 (Conv 2, Conv 3)
    data_t conv_weight2[CH][1][3][3], data_t* conv_bias2,
    data_t conv_weight3[CH][CH][1][1], data_t* conv_bias3,
    // Node 2.0 (Conv 4, Conv 5, Conv 6, Conv 7)
    data_t conv_weight4[CH][1][5][5], data_t* conv_bias4,
    data_t conv_weight5[CH][CH][1][1], data_t* conv_bias5,
    data_t conv_weight6[CH][1][5][5], data_t* conv_bias6,
    data_t conv_weight7[CH][CH][1][1], data_t* conv_bias7,
    // Node 2.1 (Conv 8, Conv 9)
    data_t conv_weight8[CH][1][5][5], data_t* conv_bias8,
    data_t conv_weight9[CH][CH][1][1], data_t* conv_bias9,
    // Node 3.0 (Conv 10, Conv 11, Conv 12, Conv 13)
    data_t conv_weight10[CH][1][5][5], data_t* conv_bias10,
    data_t conv_weight11[CH][CH][1][1], data_t* conv_bias11,
    data_t conv_weight12[CH][1][5][5], data_t* conv_bias12,
    data_t conv_weight13[CH][CH][1][1], data_t* conv_bias13
);

// 통합된 Cell 레이어 템플릿 함수 정의
template<int CH, int IN_H, int IN_W>
void cell_layer(
    // 입력 텐서들 (preprocess 출력 두 개) 
    data_t inputA[CH][IN_H][IN_W],
    data_t inputB[CH][IN_H][IN_W],
    // 출력 텐서 (4*CH 채널, 크기는 절반) 
    data_t output[4*CH][IN_H/2][IN_W/2],
    // 각 Conv 연산의 가중치와 바이어스 파라미터들
    // Node 0.1 (Conv 0, Conv 1)
    data_t conv_weight0[CH][1][5][5], data_t* conv_bias0,
    data_t conv_weight1[CH][CH][1][1], data_t* conv_bias1,
    // Node 1.1 (Conv 2, Conv 3)
    data_t conv_weight2[CH][1][3][3], data_t* conv_bias2,
    data_t conv_weight3[CH][CH][1][1], data_t* conv_bias3,
    // Node 2.0 (Conv 4, Conv 5, Conv 6, Conv 7)
    data_t conv_weight4[CH][1][5][5], data_t* conv_bias4,
    data_t conv_weight5[CH][CH][1][1], data_t* conv_bias5,
    data_t conv_weight6[CH][1][5][5], data_t* conv_bias6,
    data_t conv_weight7[CH][CH][1][1], data_t* conv_bias7,
    // Node 2.1 (Conv 8, Conv 9)
    data_t conv_weight8[CH][1][5][5], data_t* conv_bias8,
    data_t conv_weight9[CH][CH][1][1], data_t* conv_bias9,
    // Node 3.0 (Conv 10, Conv 11, Conv 12, Conv 13)
    data_t conv_weight10[CH][1][5][5], data_t* conv_bias10,
    data_t conv_weight11[CH][CH][1][1], data_t* conv_bias11,
    data_t conv_weight12[CH][1][5][5], data_t* conv_bias12,
    data_t conv_weight13[CH][CH][1][1], data_t* conv_bias13
) {
    // ===== Node 0 =====
    data_t node_0_0_maxpool_output[CH][IN_H/2][IN_W/2];
    data_t node_0_1_op0_relu_output[CH][IN_H][IN_W];
    data_t node_0_1_op1_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_0_1_op2_conv_output[CH][IN_H/2][IN_W/2];
    data_t node0_add_output[CH][IN_H/2][IN_W/2];

    // Node 0.0: MaxPooling (입력B에 3x3 풀링, stride 2)
    maxpool_layer<CH, 3, IN_H, IN_W, 2>(inputB, node_0_0_maxpool_output, 2, 1);
    // Node 0.1: ReLU -> Depthwise Conv(5x5) -> Pointwise Conv(1x1) -> Add
    relu_layer<CH, IN_H, IN_W>(inputA, node_0_1_op0_relu_output);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        node_0_1_op0_relu_output, conv_weight0, conv_bias0, 
        node_0_1_op1_conv_output, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_0_1_op1_conv_output, conv_weight1, conv_bias1, 
        node_0_1_op2_conv_output, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(
        node_0_0_maxpool_output, node_0_1_op2_conv_output, node0_add_output);

    // ===== Node 1 =====
    data_t node_1_0_maxpool_output[CH][IN_H/2][IN_W/2];
    data_t node_1_1_op1_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_1_1_op2_conv_output[CH][IN_H/2][IN_W/2];
    data_t node1_add_output[CH][IN_H/2][IN_W/2];

    // Node 1.0: MaxPooling (이전 add 출력에 3x3 풀링, stride 1)
    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node0_add_output, node_1_0_maxpool_output, 1, 1);
    // Node 1.1: Depthwise Conv(3x3) -> Pointwise Conv(1x1) -> Add
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 3, CH>(
        node_0_1_op0_relu_output, conv_weight2, conv_bias2, 
        node_1_1_op1_conv_output, 2, 2, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_1_1_op1_conv_output, conv_weight3, conv_bias3, 
        node_1_1_op2_conv_output, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(
        node_1_0_maxpool_output, node_1_1_op2_conv_output, node1_add_output);

    // ===== Node 2 =====
    data_t node_2_0_op0_relu_output[CH][IN_H/2][IN_W/2];
    data_t node_2_0_op1_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_2_0_op2_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_2_0_op4_relu_output[CH][IN_H/2][IN_W/2];
    data_t node_2_0_op5_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_2_0_op6_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_2_1_op0_relu_output[CH][IN_H][IN_W];
    data_t node_2_1_op1_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_2_1_op2_conv_output[CH][IN_H/2][IN_W/2];
    data_t node2_add_output[CH][IN_H/2][IN_W/2];

    // Node 2.0: (첫 번째 Branch) ReLU -> Depthwise Conv(5x5) -> Pointwise Conv -> ReLU -> Depthwise Conv -> Pointwise Conv
    relu_layer<CH, IN_H/2, IN_W/2>(node0_add_output, node_2_0_op0_relu_output);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        node_2_0_op0_relu_output, conv_weight4, conv_bias4, 
        node_2_0_op1_conv_output, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_2_0_op1_conv_output, conv_weight5, conv_bias5, 
        node_2_0_op2_conv_output, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(node_2_0_op2_conv_output, node_2_0_op4_relu_output);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        node_2_0_op4_relu_output, conv_weight6, conv_bias6, 
        node_2_0_op5_conv_output, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_2_0_op5_conv_output, conv_weight7, conv_bias7, 
        node_2_0_op6_conv_output, 1, 0, 1, 1);

    // Node 2.1: (두 번째 Branch) ReLU -> Depthwise Conv(5x5) -> Pointwise Conv
    relu_layer<CH, IN_H, IN_W>(inputB, node_2_1_op0_relu_output);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        node_2_1_op0_relu_output, conv_weight8, conv_bias8, 
        node_2_1_op1_conv_output, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_2_1_op1_conv_output, conv_weight9, conv_bias9, 
        node_2_1_op2_conv_output, 1, 0, 1, 1);
    // Node 2: Add (두 개 branch의 출력 합산)
    add_layer<CH, IN_H/2, IN_W/2>(
        node_2_0_op6_conv_output, node_2_1_op2_conv_output, node2_add_output);

    // ===== Node 3 =====
    data_t node_3_0_op1_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_3_0_op2_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_3_0_op4_relu_output[CH][IN_H/2][IN_W/2];
    data_t node_3_0_op5_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_3_0_op6_conv_output[CH][IN_H/2][IN_W/2];
    data_t node_3_1_maxpool_output[CH][IN_H/2][IN_W/2];
    data_t node3_add_output[CH][IN_H/2][IN_W/2];

    // Node 3.0: Depthwise Conv(5x5) -> Pointwise Conv -> ReLU -> Depthwise Conv -> Pointwise Conv
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        node_2_1_op0_relu_output, conv_weight10, conv_bias10, 
        node_3_0_op1_conv_output, 2, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_3_0_op1_conv_output, conv_weight11, conv_bias11, 
        node_3_0_op2_conv_output, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(node_3_0_op2_conv_output, node_3_0_op4_relu_output);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        node_3_0_op4_relu_output, conv_weight12, conv_bias12, 
        node_3_0_op5_conv_output, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        node_3_0_op5_conv_output, conv_weight13, conv_bias13, 
        node_3_0_op6_conv_output, 1, 0, 1, 1);

    // Node 3.1: MaxPooling -> Add
    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node2_add_output, node_3_1_maxpool_output, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(
        node_3_0_op6_conv_output, node_3_1_maxpool_output, node3_add_output);

    // 최종 출력 Concatenation (4개 브랜치 출력 병합)
    concat_layer<CH, IN_H/2, IN_W/2>(
        node0_add_output, node1_add_output, node2_add_output, node3_add_output, output);
}
