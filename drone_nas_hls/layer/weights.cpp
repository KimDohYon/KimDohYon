#include "weights.h"

// Stem 레이어 가중치
data_t stem_conv_weight[12][2][3][3];
data_t stem_conv_bias[12];

// Cell 0 레이어 가중치
data_t cell0_conv1_weight[8][12][1][1];
data_t cell0_conv1_bias[8];
data_t cell0_conv2_weight[8][12][1][1];
data_t cell0_conv2_bias[8];
data_t cell0_conv3_weight[8][8][1][1];
data_t cell0_conv3_bias[8];
data_t cell0_conv4_weight[8][8][1][1];
data_t cell0_conv4_bias[8];
data_t cell0_conv5_weight[8][8][1][1];
data_t cell0_conv5_bias[8];
data_t cell0_conv6_weight[8][8][1][1];
data_t cell0_conv6_bias[8];
data_t cell0_conv7_weight[8][8][1][1];
data_t cell0_conv7_bias[8];
data_t cell0_conv8_weight[8][8][1][1];
data_t cell0_conv8_bias[8];
data_t cell0_conv9_weight[8][8][1][1];
data_t cell0_conv9_bias[8];

// Cell 1 레이어 가중치
data_t cell1_conv1_weight[16][32][1][1];
data_t cell1_conv1_bias[16];
data_t cell1_conv2_weight[16][16][1][1];
data_t cell1_conv2_bias[16];
data_t cell1_conv3_weight[16][16][1][1];
data_t cell1_conv3_bias[16];
data_t cell1_conv4_weight[16][16][1][1];
data_t cell1_conv4_bias[16];
data_t cell1_conv5_weight[16][16][1][1];
data_t cell1_conv5_bias[16];
data_t cell1_conv6_weight[16][16][1][1];
data_t cell1_conv6_bias[16];
data_t cell1_conv7_weight[16][16][1][1];
data_t cell1_conv7_bias[16];
data_t cell1_conv8_weight[16][16][1][1];
data_t cell1_conv8_bias[16];

// Cell 1 BatchNorm 가중치
data_t cells_1_preprocess0_bn_weight[16];
data_t cells_1_preprocess0_bn_bias[16];
data_t cells_1_preprocess0_bn_running_mean[16];
data_t cells_1_preprocess0_bn_running_var[16];

// Classifier 레이어 가중치
data_t classifier_weight[12][64];
data_t classifier_bias[12];

// 추가 가중치
data_t cells_0_ops_by_node_0_1_op_1_weight[8][1][5][5];
data_t cells_0_ops_by_node_1_1_op_1_weight[8][1][3][3];
data_t cells_0_ops_by_node_2_0_op_1_weight[8][1][5][5];
data_t cells_0_ops_by_node_2_0_op_5_weight[8][1][5][5];
data_t cells_0_ops_by_node_2_1_op_1_weight[8][1][5][5];
data_t cells_0_ops_by_node_3_0_op_1_weight[8][1][5][5];
data_t cells_0_ops_by_node_3_0_op_5_weight[8][1][5][5];
data_t cells_1_preprocess0_conv1_weight[8][12][1][1];
data_t cells_1_preprocess0_conv2_weight[8][12][1][1];
data_t cells_1_ops_by_node_0_1_op_1_weight[16][1][5][5];
data_t cells_1_ops_by_node_1_1_op_1_weight[16][1][3][3];
data_t cells_1_ops_by_node_2_0_op_1_weight[16][1][5][5];
data_t cells_1_ops_by_node_2_0_op_5_weight[16][1][5][5];
data_t cells_1_ops_by_node_2_1_op_1_weight[16][1][5][5];
data_t cells_1_ops_by_node_3_0_op_1_weight[16][1][5][5];
data_t cells_1_ops_by_node_3_0_op_5_weight[16][1][5][5];

#ifndef __SYNTHESIS__
void initialize_weights() {
    std::cout << "Starting weight initialization..." << std::endl;

    // Stem 레이어 가중치 로드
    std::cout << "Loading stem layer weights..." << std::endl;
    load_weights_from_file("onnx_Conv_223.txt", stem_conv_weight);
    load_weights_from_file("onnx_Conv_224.txt", stem_conv_bias);

    // Cell 0 레이어 가중치 로드
    std::cout << "Loading cell0 layer weights..." << std::endl;
    load_weights_from_file("onnx_Conv_226.txt", cell0_conv1_weight);
    load_weights_from_file("onnx_Conv_227.txt", cell0_conv1_bias);
    load_weights_from_file("onnx_Conv_232.txt", cell0_conv3_weight);
    load_weights_from_file("onnx_Conv_233.txt", cell0_conv3_bias);

    // Cell 0 레이어 가중치 로드
    load_weights_from_file("onnx_Conv_229.txt", cell0_conv2_weight);
    load_weights_from_file("onnx_Conv_230.txt", cell0_conv2_bias);
    load_weights_from_file("onnx_Conv_235.txt", cell0_conv4_weight);
    load_weights_from_file("onnx_Conv_236.txt", cell0_conv4_bias);
    load_weights_from_file("onnx_Conv_238.txt", cell0_conv5_weight);
    load_weights_from_file("onnx_Conv_239.txt", cell0_conv5_bias);
    load_weights_from_file("onnx_Conv_241.txt", cell0_conv6_weight);
    load_weights_from_file("onnx_Conv_242.txt", cell0_conv6_bias);
    load_weights_from_file("onnx_Conv_244.txt", cell0_conv7_weight);
    load_weights_from_file("onnx_Conv_245.txt", cell0_conv7_bias);
    load_weights_from_file("onnx_Conv_247.txt", cell0_conv8_weight);
    load_weights_from_file("onnx_Conv_248.txt", cell0_conv8_bias);
    load_weights_from_file("onnx_Conv_250.txt", cell0_conv9_weight);
    load_weights_from_file("onnx_Conv_251.txt", cell0_conv9_bias);

    // Cell 1 레이어 가중치 로드
    load_weights_from_file("onnx_Conv_253.txt", cell1_conv1_weight);
    load_weights_from_file("onnx_Conv_254.txt", cell1_conv1_bias);
    load_weights_from_file("onnx_Conv_256.txt", cell1_conv2_weight);
    load_weights_from_file("onnx_Conv_257.txt", cell1_conv2_bias);
    load_weights_from_file("onnx_Conv_259.txt", cell1_conv3_weight);
    load_weights_from_file("onnx_Conv_260.txt", cell1_conv3_bias);
    load_weights_from_file("onnx_Conv_262.txt", cell1_conv4_weight);
    load_weights_from_file("onnx_Conv_263.txt", cell1_conv4_bias);
    load_weights_from_file("onnx_Conv_265.txt", cell1_conv5_weight);
    load_weights_from_file("onnx_Conv_266.txt", cell1_conv5_bias);
    load_weights_from_file("onnx_Conv_268.txt", cell1_conv6_weight);
    load_weights_from_file("onnx_Conv_269.txt", cell1_conv6_bias);
    load_weights_from_file("onnx_Conv_271.txt", cell1_conv7_weight);
    load_weights_from_file("onnx_Conv_272.txt", cell1_conv7_bias);
    load_weights_from_file("onnx_Conv_274.txt", cell1_conv8_weight);
    load_weights_from_file("onnx_Conv_275.txt", cell1_conv8_bias);

    // Cell 1 BatchNorm 가중치 로드
    load_weights_from_file("cells.1.preprocess0.bn.weight.txt", cells_1_preprocess0_bn_weight);
    load_weights_from_file("cells.1.preprocess0.bn.bias.txt", cells_1_preprocess0_bn_bias);
    load_weights_from_file("cells.1.preprocess0.bn.running_mean.txt", cells_1_preprocess0_bn_running_mean);
    load_weights_from_file("cells.1.preprocess0.bn.running_var.txt", cells_1_preprocess0_bn_running_var);

    // Classifier 레이어 가중치 로드
    load_weights_from_file("classifier.weight.txt", classifier_weight);
    load_weights_from_file("classifier.bias.txt", classifier_bias);

    // 추가 가중치 로드
    load_weights_from_file("cells.0.ops_by_node.0.1.op.1.weight.txt", cells_0_ops_by_node_0_1_op_1_weight);
    load_weights_from_file("cells.0.ops_by_node.1.1.op.1.weight.txt", cells_0_ops_by_node_1_1_op_1_weight);
    load_weights_from_file("cells.0.ops_by_node.2.0.op.1.weight.txt", cells_0_ops_by_node_2_0_op_1_weight);
    load_weights_from_file("cells.0.ops_by_node.2.0.op.5.weight.txt", cells_0_ops_by_node_2_0_op_5_weight);
    load_weights_from_file("cells.0.ops_by_node.2.1.op.1.weight.txt", cells_0_ops_by_node_2_1_op_1_weight);
    load_weights_from_file("cells.0.ops_by_node.3.0.op.1.weight.txt", cells_0_ops_by_node_3_0_op_1_weight);
    load_weights_from_file("cells.0.ops_by_node.3.0.op.5.weight.txt", cells_0_ops_by_node_3_0_op_5_weight);
    load_weights_from_file("cells.1.preprocess0.conv1.weight.txt", cells_1_preprocess0_conv1_weight);
    load_weights_from_file("cells.1.preprocess0.conv2.weight.txt", cells_1_preprocess0_conv2_weight);
    load_weights_from_file("cells.1.ops_by_node.0.1.op.1.weight.txt", cells_1_ops_by_node_0_1_op_1_weight);
    load_weights_from_file("cells.1.ops_by_node.1.1.op.1.weight.txt", cells_1_ops_by_node_1_1_op_1_weight);
    load_weights_from_file("cells.1.ops_by_node.2.0.op.1.weight.txt", cells_1_ops_by_node_2_0_op_1_weight);
    load_weights_from_file("cells.1.ops_by_node.2.0.op.5.weight.txt", cells_1_ops_by_node_2_0_op_5_weight);
    load_weights_from_file("cells.1.ops_by_node.2.1.op.1.weight.txt", cells_1_ops_by_node_2_1_op_1_weight);
    load_weights_from_file("cells.1.ops_by_node.3.0.op.1.weight.txt", cells_1_ops_by_node_3_0_op_1_weight);
    load_weights_from_file("cells.1.ops_by_node.3.0.op.5.weight.txt", cells_1_ops_by_node_3_0_op_5_weight);
}
#endif