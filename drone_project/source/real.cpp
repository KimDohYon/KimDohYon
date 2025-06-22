#include "model.h"
#include "relu_layer.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "maxpool_layer.h"
#include "conv_wrapper.h"

#define CH 8
#define IN_H 32
#define IN_W 64

void model_top(
    data_t inputA[CH][IN_H][IN_W],
    data_t inputB[CH][IN_H][IN_W],
    data_t output[4*CH][IN_H/2][IN_W/2],

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
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
#pragma HLS INTERFACE m_axi port=inputA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=inputB offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

#pragma HLS INTERFACE m_axi port=conv_weight0 offset=slave bundle=weights_mem0 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias0 offset=slave bundle=weights_mem0 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight1 offset=slave bundle=weights_mem0 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias1 offset=slave bundle=weights_mem0 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight2 offset=slave bundle=weights_mem0 depth=72
#pragma HLS INTERFACE m_axi port=conv_bias2 offset=slave bundle=weights_mem0 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight3 offset=slave bundle=weights_mem0 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias3 offset=slave bundle=weights_mem0 depth=8

#pragma HLS INTERFACE m_axi port=conv_weight4 offset=slave bundle=weights_mem1 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias4 offset=slave bundle=weights_mem1 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight5 offset=slave bundle=weights_mem1 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias5 offset=slave bundle=weights_mem1 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight6 offset=slave bundle=weights_mem1 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias6 offset=slave bundle=weights_mem1 depth=8

#pragma HLS INTERFACE m_axi port=conv_weight7 offset=slave bundle=weights_mem2 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias7 offset=slave bundle=weights_mem2 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight8 offset=slave bundle=weights_mem2 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias8 offset=slave bundle=weights_mem2 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight9 offset=slave bundle=weights_mem2 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias9 offset=slave bundle=weights_mem2 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight10 offset=slave bundle=weights_mem2 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias10 offset=slave bundle=weights_mem2 depth=8

#pragma HLS INTERFACE m_axi port=conv_weight11 offset=slave bundle=weights_mem3 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias11 offset=slave bundle=weights_mem3 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight12 offset=slave bundle=weights_mem3 depth=200
#pragma HLS INTERFACE m_axi port=conv_bias12 offset=slave bundle=weights_mem3 depth=8
#pragma HLS INTERFACE m_axi port=conv_weight13 offset=slave bundle=weights_mem3 depth=64
#pragma HLS INTERFACE m_axi port=conv_bias13 offset=slave bundle=weights_mem3 depth=8

    // Buffer Reuse
    data_t reuse_buffer0[CH][IN_H/2][IN_W/2];
    data_t reuse_buffer1[CH][IN_H/2][IN_W/2];
    data_t reuse_buffer2[CH][IN_H/2][IN_W/2];
    data_t reuse_buffer3[CH][IN_H][IN_W];

    data_t node0_add_output[CH][IN_H/2][IN_W/2];
    data_t node1_add_output[CH][IN_H/2][IN_W/2];
    data_t node2_add_output[CH][IN_H/2][IN_W/2];
    data_t node3_add_output[CH][IN_H/2][IN_W/2];

    // ===== Node 0 =====
    maxpool_layer<CH, 3, IN_H, IN_W, 2>(inputB, reuse_buffer0, 2, 1);
    relu_layer<CH, IN_H, IN_W>(inputA, reuse_buffer3);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer3, conv_weight0, conv_bias0, reuse_buffer1, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer1, conv_weight1, conv_bias1, reuse_buffer2, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(reuse_buffer0, reuse_buffer2, node0_add_output); 

    // ===== Node 1 =====
    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node0_add_output, reuse_buffer0, 1, 1);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 3, CH>(
        reuse_buffer3, conv_weight2, conv_bias2, reuse_buffer1, 2, 2, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer1, conv_weight3, conv_bias3, reuse_buffer2, 1, 0, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(reuse_buffer0, reuse_buffer2, node1_add_output);

    // ===== Node 2 =====
    relu_layer<CH, IN_H/2, IN_W/2>(node0_add_output, reuse_buffer0);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer0, conv_weight4, conv_bias4, reuse_buffer1, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer1, conv_weight5, conv_bias5, reuse_buffer0, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(reuse_buffer0, reuse_buffer1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer1, conv_weight6, conv_bias6, reuse_buffer0, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer0, conv_weight7, conv_bias7, reuse_buffer1, 1, 0, 1, 1);

    relu_layer<CH, IN_H, IN_W>(inputB, reuse_buffer3);
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer3, conv_weight8, conv_bias8, reuse_buffer0, 2, 4, CH, 2);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer0, conv_weight9, conv_bias9, reuse_buffer2, 1, 0, 1, 1);

    add_layer<CH, IN_H/2, IN_W/2>(reuse_buffer1, reuse_buffer2, node2_add_output);

    // ===== Node 3 =====
    conv_layer<CH, CH, IN_H, IN_W, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer3, conv_weight10, conv_bias10, reuse_buffer0, 2, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer0, conv_weight11, conv_bias11, reuse_buffer1, 1, 0, 1, 1);
    relu_layer<CH, IN_H/2, IN_W/2>(reuse_buffer1, reuse_buffer0);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 5, CH>(
        reuse_buffer0, conv_weight12, conv_bias12, reuse_buffer1, 1, 2, CH, 1);
    conv_layer<CH, CH, IN_H/2, IN_W/2, IN_H/2, IN_W/2, 1, 1>(
        reuse_buffer1, conv_weight13, conv_bias13, reuse_buffer0, 1, 0, 1, 1);

    maxpool_layer<CH, 3, IN_H/2, IN_W/2, 1>(node2_add_output, reuse_buffer1, 1, 1);
    add_layer<CH, IN_H/2, IN_W/2>(reuse_buffer0, reuse_buffer1, node3_add_output);

    // Concatenation
    concat_layer<CH, IN_H/2, IN_W/2>(
        node0_add_output, node1_add_output, node2_add_output, node3_add_output, output);
}
