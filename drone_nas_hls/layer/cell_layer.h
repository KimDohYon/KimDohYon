// DATAFLOW 및 BRAM 최적화 적용된 cell_layer.h
#pragma once
#include "model.h"
#include "relu_layer.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "maxpool_layer.h"
#include "conv_wrapper.h"

// Note: conv_layer는 재사용 가능한 중간 buffer를 공유하도록 개선
// 또한, ARRAY_PARTITION 대신 LINE_BUFFER 형태로 처리 가능 시 LUTRAM 유도
// 각 노드 내부 pipeline 유지 + 공유 buffer 최소화로 BRAM 사용 줄임

// ===== Node 0 정의 =====
template<int CH, int IN_H, int IN_W>
void node0(
    data_t inputA[CH][IN_H][IN_W],
    data_t inputB[CH][IN_H][IN_W],
    data_t conv_weight0[CH][1][5][5], data_t* conv_bias0,
    data_t conv_weight1[CH][CH][1][1], data_t* conv_bias1,
    data_t node0_add_output[CH][IN_H/2][IN_W/2]
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static data_t relu_out[CH][IN_H][IN_W];
    static data_t inter_out[CH][IN_H/2][IN_W/2];
    static data_t max_out[CH][IN_H/2][IN_W/2];

    relu_layer<CH, IN_H, IN_W>(inputA, relu_out);
    maxpool_layer<CH, 3, IN_H, IN_W, 2>(inputB, max_out, 2, 1);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(relu_out, conv_weight0, conv_bias0, inter_out, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(inter_out, conv_weight1, conv_bias1, inter_out, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(max_out, inter_out, node0_add_output);
}

// ===== Node 1 정의 =====
template<int CH, int IN_H, int IN_W>
void node1(
    data_t node0_out[CH][IN_H/2][IN_W/2],
    data_t inputA[CH][IN_H][IN_W],
    data_t conv_weight2[CH][1][3][3], data_t* conv_bias2,
    data_t conv_weight3[CH][CH][1][1], data_t* conv_bias3,
    data_t node1_out[CH][IN_H/2][IN_W/2]
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static data_t relu_out[CH][IN_H][IN_W];
    static data_t conv_out1[CH][IN_H/2][IN_W/2];
    static data_t conv_out2[CH][IN_H/2][IN_W/2];
    static data_t max_out[CH][IN_H/2][IN_W/2];

    relu_layer<CH, IN_H, IN_W>(inputA, relu_out);
    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node0_out, max_out, 1, 1);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 3, CH>(relu_out, conv_weight2, conv_bias2, conv_out1, 2, 2, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(conv_out1, conv_weight3, conv_bias3, conv_out2, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(max_out, conv_out2, node1_out);
}

// ===== Node 2 정의 =====
template<int CH, int IN_H, int IN_W>
void node2(
    data_t node0_out[CH][IN_H/2][IN_W/2],
    data_t inputB[CH][IN_H][IN_W],
    data_t conv_weight4[CH][1][5][5], data_t* conv_bias4,
    data_t conv_weight5[CH][CH][1][1], data_t* conv_bias5,
    data_t conv_weight6[CH][1][5][5], data_t* conv_bias6,
    data_t conv_weight7[CH][CH][1][1], data_t* conv_bias7,
    data_t conv_weight8[CH][1][5][5], data_t* conv_bias8,
    data_t conv_weight9[CH][CH][1][1], data_t* conv_bias9,
    data_t node2_out[CH][IN_H/2][IN_W/2]
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static data_t r0[CH][IN_H/2][IN_W/2], c0[CH][IN_H/2][IN_W/2], c1[CH][IN_H/2][IN_W/2];
    static data_t r1[CH][IN_H/2][IN_W/2], c2[CH][IN_H/2][IN_W/2], c3[CH][IN_H/2][IN_W/2];
    static data_t r2[CH][IN_H][IN_W], c4[CH][IN_H/2][IN_W/2], c5[CH][IN_H/2][IN_W/2];

    relu_layer<CH, IN_H/2, IN_W/2>(node0_out, r0);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(r0, conv_weight4, conv_bias4, c0, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(c0, conv_weight5, conv_bias5, c1, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(c1, r1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(r1, conv_weight6, conv_bias6, c2, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(c2, conv_weight7, conv_bias7, c3, 1, 0, 1, 1);
    relu_layer<CH, IN_H, IN_W>(inputB, r2);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(r2, conv_weight8, conv_bias8, c4, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(c4, conv_weight9, conv_bias9, c5, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(c3, c5, node2_out);
}

// ===== Node 3 정의 =====
template<int CH, int IN_H, int IN_W>
void node3(
    data_t node2_out[CH][IN_H/2][IN_W/2],
    data_t inputB[CH][IN_H][IN_W],
    data_t conv_weight10[CH][1][5][5], data_t* conv_bias10,
    data_t conv_weight11[CH][CH][1][1], data_t* conv_bias11,
    data_t conv_weight12[CH][1][5][5], data_t* conv_bias12,
    data_t conv_weight13[CH][CH][1][1], data_t* conv_bias13,
    data_t node3_out[CH][IN_H/2][IN_W/2]
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static data_t c0[CH][IN_H/2][IN_W/2], c1[CH][IN_H/2][IN_W/2];
    static data_t r1[CH][IN_H/2][IN_W/2], c2[CH][IN_H/2][IN_W/2];
    static data_t c3[CH][IN_H/2][IN_W/2], mp[CH][IN_H/2][IN_W/2];

    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(inputB, conv_weight10, conv_bias10, c0, 2, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(c0, conv_weight11, conv_bias11, c1, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(c1, r1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(r1, conv_weight12, conv_bias12, c2, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(c2, conv_weight13, conv_bias13, c3, 1, 0, 1, 1);
    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node2_out, mp, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(c3, mp, node3_out);
}

template<int CH, int IN_H, int IN_W>
void cell_layer(
    data_t inputA[CH][IN_H][IN_W],
    data_t inputB[CH][IN_H][IN_W],
    data_t output[4 * CH][IN_H / 2][IN_W / 2],
    data_t conv_weight0[CH][1][5][5], data_t* conv_bias0,
    data_t conv_weight1[CH][CH][1][1], data_t* conv_bias1,
    data_t conv_weight2[CH][1][3][3], data_t* conv_bias2,
    data_t conv_weight3[CH][CH][1][1], data_t* conv_bias3,
    data_t conv_weight4[CH][1][5][5], data_t* conv_bias4,
    data_t conv_weight5[CH][CH][1][1], data_t* conv_bias5,
    data_t conv_weight6[CH][1][5][5], data_t* conv_bias6,
    data_t conv_weight7[CH][CH][1][1], data_t* conv_bias7,
    data_t conv_weight8[CH][1][5][5], data_t* conv_bias8,
    data_t conv_weight9[CH][CH][1][1], data_t* conv_bias9,
    data_t conv_weight10[CH][1][5][5], data_t* conv_bias10,
    data_t conv_weight11[CH][CH][1][1], data_t* conv_bias11,
    data_t conv_weight12[CH][1][5][5], data_t* conv_bias12,
    data_t conv_weight13[CH][CH][1][1], data_t* conv_bias13
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

data_t node0_out[CH][IN_H / 2][IN_W / 2];
#pragma HLS STREAM variable=node0_out depth=1

data_t node1_out[CH][IN_H / 2][IN_W / 2];
#pragma HLS STREAM variable=node1_out depth=1

data_t node2_out[CH][IN_H / 2][IN_W / 2];
#pragma HLS STREAM variable=node2_out depth=1

data_t node3_out[CH][IN_H / 2][IN_W / 2];
#pragma HLS STREAM variable=node3_out depth=1


    node0<CH, IN_H, IN_W>(inputA, inputB, conv_weight0, conv_bias0, conv_weight1, conv_bias1, node0_out);
    node1<CH, IN_H, IN_W>(node0_out, inputA, conv_weight2, conv_bias2, conv_weight3, conv_bias3, node1_out);
    node2<CH, IN_H, IN_W>(node0_out, inputB,
                         conv_weight4, conv_bias4, conv_weight5, conv_bias5,
                         conv_weight6, conv_bias6, conv_weight7, conv_bias7,
                         conv_weight8, conv_bias8, conv_weight9, conv_bias9,
                         node2_out);
    node3<CH, IN_H, IN_W>(node2_out, inputB,
                         conv_weight10, conv_bias10, conv_weight11, conv_bias11,
                         conv_weight12, conv_bias12, conv_weight13, conv_bias13,
                         node3_out);

    concat_layer<CH, IN_H / 2, IN_W / 2>(node0_out, node1_out, node2_out, node3_out, output);
}