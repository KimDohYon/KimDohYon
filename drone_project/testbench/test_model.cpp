#include "model.h"
#include "weights.h"
#include "cell_layer.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "maxpool_layer.h"
#include "global_pooling.h"
#include "add_layer.h"
#include "concat_layer.h"
#include "batchnorm_layer.h"
#include "gemm_layer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdint>
void save_3d_tensor(const char* filename, data_t tensor[][CELL0_IN_HEIGHT][CELL0_IN_WIDTH], int channels, int height, int width) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }
    for (int c = 0; c < channels; c++) {
        file << "Channel " << c << ":\n";
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                file << tensor[c][h][w] << " ";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}
void save_3d_tensor_cell0_out(const char* filename, data_t tensor[][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2], int channels, int height, int width) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }
    for (int c = 0; c < channels; c++) {
        file << "Channel " << c << ":\n";
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                file << tensor[c][h][w] << " ";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}

void stem_layer(
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]
) {
    conv_layer<IN_CHANNELS, STEM_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 3, 1>(
        input, stem_conv_weight, stem_conv_bias, stem0_conv_output, 1, 1, 1, 1
    );
}

void read_input_data(const char* filename, data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "입력 파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }
    for (int c = 0; c < IN_CHANNELS; c++) {
        for (int h = 0; h < CELL0_IN_HEIGHT; h++) {
            for (int w = 0; w < CELL0_IN_WIDTH; w++) {
                int16_t value;
                file.read(reinterpret_cast<char*>(&value), sizeof(int16_t));
                input[c][h][w] = static_cast<float>(value);
            }
        }
    }
    file.close();
}

void print_inference_results(data_t output[OUT_CHANNELS]) {
    float max_val = -std::numeric_limits<float>::infinity();
    float min_val = std::numeric_limits<float>::infinity();
    float max_prob = -1.0f;
    int max_class = -1;
    float sum_exp = 0.0f;
    float probs[OUT_CHANNELS];

    for (int i = 0; i < OUT_CHANNELS; ++i) {
        max_val = std::max(max_val, output[i]);
        min_val = std::min(min_val, output[i]);
    }

    for (int i = 0; i < OUT_CHANNELS; ++i) {
        probs[i] = std::exp(output[i] - max_val);
        sum_exp += probs[i];
    }

    std::cout << "=== 클래스별 확률 출력 ===" << std::endl;
    for (int i = 0; i < OUT_CHANNELS; ++i) {
        float prob = probs[i] / sum_exp;
        std::cout << "클래스 " << i << ": " << std::fixed << std::setprecision(4) << prob * 100 << "%" << std::endl;
        if (prob > max_prob) {
            max_prob = prob;
            max_class = i;
        }
    }

    std::cout << "최종 예측: 클래스 " << max_class << " (확률: " << std::fixed << std::setprecision(2) << max_prob * 100 << "%)" << std::endl;
    std::cout << "출력 통계: 최소값 = " << min_val << ", 최대값 = " << max_val << std::endl;
}

int main() {
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t output[OUT_CHANNELS];

    // 중간 버퍼
    data_t stem_out[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t pre0_relu[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t pre0_conv[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t pre1_conv[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell0_out[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2];

    data_t cell1_pre0_bn[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t cell1_pre1_conv[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t cell1_out[4*CELL1_CHANNELS][CELL1_IN_HEIGHT/2][CELL1_IN_WIDTH/2];
    data_t pooled[4*CELL1_CHANNELS];

    initialize_weights();
    read_input_data("test.bin", input);

    // === stem & cell0 preprocess ===
    stem_layer(input, stem_out);
    relu_layer<STEM_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH>(stem_out, pre0_relu);
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 1, 1>(
        pre0_relu, cell0_conv1_weight, cell0_conv1_bias, pre0_conv, 1, 0, 1, 1);
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, 1, 1>(
        pre0_relu, cell0_conv2_weight, cell0_conv2_bias, pre1_conv, 1, 0, 1, 1);
    
    save_3d_tensor("pre0_conv.txt", pre0_conv, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH);
    save_3d_tensor("pre1_conv.txt", pre1_conv, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH);

    // === cell0
    model_top(
        pre0_conv, pre1_conv, cell0_out,
        cells_0_ops_by_node_0_1_op_1_weight, nullptr,
        cell0_conv3_weight, cell0_conv3_bias,
        cells_0_ops_by_node_1_1_op_1_weight, nullptr,
        cell0_conv4_weight, cell0_conv4_bias,
        cells_0_ops_by_node_2_0_op_1_weight, nullptr,
        cell0_conv5_weight, cell0_conv5_bias,
        cells_0_ops_by_node_2_0_op_5_weight, nullptr,
        cell0_conv6_weight, cell0_conv6_bias,
        cells_0_ops_by_node_2_1_op_1_weight, nullptr,
        cell0_conv7_weight, cell0_conv7_bias,
        cells_0_ops_by_node_3_0_op_1_weight, nullptr,
        cell0_conv8_weight, cell0_conv8_bias,
        cells_0_ops_by_node_3_0_op_5_weight, nullptr,
        cell0_conv9_weight, cell0_conv9_bias
    );
    save_3d_tensor_cell0_out("cell0_out.txt", cell0_out, 4*CELL0_CHANNELS, CELL0_IN_HEIGHT/2, CELL0_IN_WIDTH/2);

    // === cell1 preprocess
    // preprocess0
    data_t pre0_conv1[CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t pre0_conv2[CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    data_t pre0_concat[2*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL1_IN_HEIGHT, CELL1_IN_WIDTH, 1, 1>(
        pre0_relu, cells_1_preprocess0_conv1_weight, nullptr, pre0_conv1, 2, 0, 1, 1);
    conv_layer<STEM_CHANNELS, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH, CELL1_IN_HEIGHT, CELL1_IN_WIDTH, 1, 1>(
        pre0_relu, cells_1_preprocess0_conv2_weight, nullptr, pre0_conv2, 2, 0, 1, 1);
    concat2_layer<CELL0_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH>(pre0_conv1, pre0_conv2, pre0_concat);
    batchnorm_layer<2*CELL0_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH>(pre0_concat, cells_1_preprocess0_bn_weight,
        cells_1_preprocess0_bn_bias, cells_1_preprocess0_bn_running_mean, cells_1_preprocess0_bn_running_var, cell1_pre0_bn);

    // preprocess1
    data_t cell1_pre1_relu[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH];
    relu_layer<4*CELL0_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH>(cell0_out, cell1_pre1_relu);
    conv_layer<4*CELL0_CHANNELS, CELL1_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH, CELL1_IN_HEIGHT, CELL1_IN_WIDTH, 1, 1>(
        cell1_pre1_relu, cell1_conv1_weight, cell1_conv1_bias, cell1_pre1_conv, 1, 0, 1, 1);

    // === cell1
    cell_layer<CELL1_CHANNELS, CELL1_IN_HEIGHT, CELL1_IN_WIDTH>(
        cell1_pre0_bn, cell1_pre1_conv, cell1_out,
        cells_1_ops_by_node_0_1_op_1_weight, nullptr,
        cell1_conv2_weight, cell1_conv2_bias,
        cells_1_ops_by_node_1_1_op_1_weight, nullptr,
        cell1_conv3_weight, cell1_conv3_bias,
        cells_1_ops_by_node_2_0_op_1_weight, nullptr,
        cell1_conv4_weight, cell1_conv4_bias,
        cells_1_ops_by_node_2_0_op_5_weight, nullptr,
        cell1_conv5_weight, cell1_conv5_bias,
        cells_1_ops_by_node_2_1_op_1_weight, nullptr,
        cell1_conv6_weight, cell1_conv6_bias,
        cells_1_ops_by_node_3_0_op_1_weight, nullptr,
        cell1_conv7_weight, cell1_conv7_bias,
        cells_1_ops_by_node_3_0_op_5_weight, nullptr,
        cell1_conv8_weight, cell1_conv8_bias
    );

    // === Global Pool + FC
    global_avg_pool<4*CELL1_CHANNELS, CELL1_IN_HEIGHT/2, CELL1_IN_WIDTH/2>(cell1_out, pooled);
    gemm_layer_1d<STEM_CHANNELS, 4*CELL1_CHANNELS>(pooled, classifier_weight, classifier_bias, output);

    print_inference_results(output);
    return 0;
}