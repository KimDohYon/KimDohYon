#include "model.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "maxpool_layer.h"
#include "global_pooling.h"
#include "weights.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "batchnorm_layer.h"
#include "cell_layer.h"

#include "gemm_layer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdint>

void classifier_layer(
    data_t input[4*CELL1_CHANNELS],
    data_t output[STEM_CHANNELS]
) {
    // Fully Connected 레이어 실행
    gemm_layer_1d<STEM_CHANNELS, 4*CELL1_CHANNELS>(
        input, classifier_weight, classifier_bias, output
    );
}

void stem_layer(
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]
) {
    conv_layer<IN_CHANNELS, STEM_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 3, 1>(
        input, stem_conv_weight, stem_conv_bias, stem0_conv_output, 1, 1, 1, 1
    );
}

void cell0_preprocess(
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre0_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre1_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre0_op0_relu_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]
) {
    //preprocess0
    relu_layer<STEM_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH>(stem0_conv_output, pre0_op0_relu_output);
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 1, 1>(
        pre0_op0_relu_output, cell0_conv1_weight, cell0_conv1_bias, pre0_op1_conv_output, 1, 0, 1, 1);
    //preprocess1
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 1, 1>(
        pre0_op0_relu_output, cell0_conv2_weight, cell0_conv2_bias, pre1_op1_conv_output, 1, 0, 1, 1);
}



void cell1_preprocess(
    data_t pre0_op0_relu_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t concat_output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2],
    data_t pre0_bn_batchnorm_output[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH],
    data_t pre1_op1_conv_output[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH]
) {
    // ===== Preprocess0 =====
    data_t pre0_conv1_output[CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t pre0_conv2_output[CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t pre0_concat_output[2*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];

    // ===== Preprocess1 =====
    data_t pre1_op0_relu_output[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];

    //preprocess0
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2, 1, 1>(
        pre0_op0_relu_output, cells_1_preprocess0_conv1_weight, nullptr, pre0_conv1_output, 2, 0, 1, 1);
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2, 1, 1>(
        pre0_op0_relu_output, cells_1_preprocess0_conv2_weight, nullptr, pre0_conv2_output, 2, 0, 1, 1);
    concat2_layer<CELL0_CHANNELS, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2>(
        pre0_conv1_output, pre0_conv2_output, pre0_concat_output);
    batchnorm_layer<2*CELL0_CHANNELS, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2>(
        pre0_concat_output, cells_1_preprocess0_bn_weight, cells_1_preprocess0_bn_bias, cells_1_preprocess0_bn_running_mean, cells_1_preprocess0_bn_running_var, pre0_bn_batchnorm_output);
    //preprocess1
    relu_layer<4*CELL0_CHANNELS, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2>(
        concat_output, pre1_op0_relu_output);
    conv_layer<4*CELL0_CHANNELS, CELL1_CHANNELS, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2, 1, 1>(
        pre1_op0_relu_output, cell1_conv1_weight, cell1_conv1_bias, pre1_op1_conv_output, 1, 0, 1, 1);

}

void model_top(
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t output[OUT_CHANNELS]
) {
    // 중간 결과를 저장할 버퍼
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell0_pre0_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell0_pre1_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell0_concat_output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2];
    data_t cell0_pre0_op0_relu_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell1_pre0_concat_output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2];
    data_t cell1_pre0_bn_batchnorm_output[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t cell1_pre1_op1_conv_output[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t cell1_concat_output[4*CELL1_CHANNELS][CELL1_IN_HEIGHT/2][CELL1_IN_WIDTH/2];
    data_t pooled_output[4*CELL1_CHANNELS];
    
    stem_layer(input, stem0_conv_output);
    cell0_preprocess(stem0_conv_output,  cell0_pre0_op1_conv_output, cell0_pre1_op1_conv_output, cell0_pre0_op0_relu_output);
    cell_layer<CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH>(
        cell0_pre0_op1_conv_output, 
        cell0_pre1_op1_conv_output, 
        cell0_concat_output,
        // Node 0.1 가중치/바이어스
        cells_0_ops_by_node_0_1_op_1_weight, nullptr, 
        cell0_conv3_weight, cell0_conv3_bias,
        // Node 1.1 가중치/바이어스 
        cells_0_ops_by_node_1_1_op_1_weight, nullptr,
        cell0_conv4_weight, cell0_conv4_bias,
        // Node 2.0 가중치/바이어스 
        cells_0_ops_by_node_2_0_op_1_weight, nullptr,
        cell0_conv5_weight, cell0_conv5_bias,
        cells_0_ops_by_node_2_0_op_5_weight, nullptr,
        cell0_conv6_weight, cell0_conv6_bias,
        // Node 2.1 가중치/바이어스
        cells_0_ops_by_node_2_1_op_1_weight, nullptr,
        cell0_conv7_weight, cell0_conv7_bias,
        // Node 3.0 가중치/바이어스 
        cells_0_ops_by_node_3_0_op_1_weight, nullptr,
        cell0_conv8_weight, cell0_conv8_bias,
        cells_0_ops_by_node_3_0_op_5_weight, nullptr,
        cell0_conv9_weight, cell0_conv9_bias
    );
    cell1_preprocess(cell0_pre0_op0_relu_output, cell0_concat_output, cell1_pre0_bn_batchnorm_output, cell1_pre1_op1_conv_output);
    cell_layer<CELL1_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH>(
        cell1_pre0_bn_batchnorm_output, 
        cell1_pre1_op1_conv_output, 
        cell1_concat_output,
        // Node 0.1 가중치/바이어스 
        cells_1_ops_by_node_0_1_op_1_weight, nullptr,
        cell1_conv2_weight, cell1_conv2_bias,
        // Node 1.1 가중치/바이어스
        cells_1_ops_by_node_1_1_op_1_weight, nullptr,
        cell1_conv3_weight, cell1_conv3_bias,
        // Node 2.0 가중치/바이어스
        cells_1_ops_by_node_2_0_op_1_weight, nullptr,
        cell1_conv4_weight, cell1_conv4_bias,
        cells_1_ops_by_node_2_0_op_5_weight, nullptr,
        cell1_conv5_weight, cell1_conv5_bias,
        // Node 2.1 가중치/바이어스
        cells_1_ops_by_node_2_1_op_1_weight, nullptr,
        cell1_conv6_weight, cell1_conv6_bias,
        // Node 3.0 가중치/바이어스
        cells_1_ops_by_node_3_0_op_1_weight, nullptr,
        cell1_conv7_weight, cell1_conv7_bias,
        cells_1_ops_by_node_3_0_op_5_weight, nullptr,
        cell1_conv8_weight, cell1_conv8_bias
    );

    global_avg_pool<4*CELL1_CHANNELS, CELL1_IN_HEIGHT/2, CELL1_IN_WIDTH/2>(cell1_concat_output, pooled_output);
    classifier_layer(pooled_output, output);

}
