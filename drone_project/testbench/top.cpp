#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdint>
#include "model.h"
#include "weights.h"

// Load input vector from file and reshape into 3D tensor
void load_input_vector_as_tensor(const char* filename, data_t tensor[][CELL0_IN_HEIGHT][CELL0_IN_WIDTH], int channels, int height, int width) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return;
    }

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                file >> tensor[c][h][w];
            }
        }
    }

    file.close();
}

// Load output vector from file and reshape into 3D tensor
void load_output_vector_as_tensor(const char* filename, data_t tensor[][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2], int channels, int height, int width) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                file >> tensor[c][h][w];
            }
        }
    }

    file.close();
}

// Dummy bias (used when actual bias is unavailable)
data_t dummy_bias[8] = {0};

// === Main Testbench ===
int main() {
    // Allocate input and output tensors
    data_t pre0_conv[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t pre1_conv[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t cell0_out[4 * CELL0_CHANNELS][CELL0_IN_HEIGHT / 2][CELL0_IN_WIDTH / 2];
    data_t ref_cell0_out[4 * CELL0_CHANNELS][CELL0_IN_HEIGHT / 2][CELL0_IN_WIDTH / 2];

    // Load input and reference output data
    load_input_vector_as_tensor("pre0_conv_vector.txt", pre0_conv, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH);
    load_input_vector_as_tensor("pre1_conv_vector.txt", pre1_conv, CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH);
    load_output_vector_as_tensor("cell0_out_vector.txt", ref_cell0_out, 4 * CELL0_CHANNELS, CELL0_IN_HEIGHT / 2, CELL0_IN_WIDTH / 2);

    // Initialize weights
    initialize_weights();

    // Execute model_top function with actual and dummy weights
    model_top(
        pre0_conv, pre1_conv, cell0_out,
        cells_0_ops_by_node_0_1_op_1_weight, dummy_bias,
        cell0_conv3_weight, cell0_conv3_bias,
        cells_0_ops_by_node_1_1_op_1_weight, dummy_bias,
        cell0_conv4_weight, cell0_conv4_bias,
        cells_0_ops_by_node_2_0_op_1_weight, dummy_bias,
        cell0_conv5_weight, cell0_conv5_bias,
        cells_0_ops_by_node_2_0_op_5_weight, dummy_bias,
        cell0_conv6_weight, cell0_conv6_bias,
        cells_0_ops_by_node_2_1_op_1_weight, dummy_bias,
        cell0_conv7_weight, cell0_conv7_bias,
        cells_0_ops_by_node_3_0_op_1_weight, dummy_bias,
        cell0_conv8_weight, cell0_conv8_bias,
        cells_0_ops_by_node_3_0_op_5_weight, dummy_bias,
        cell0_conv9_weight, cell0_conv9_bias
    );

    // Compare output with reference
    int mismatch = 0;
    for (int c = 0; c < 4 * CELL0_CHANNELS; ++c) {
        for (int h = 0; h < CELL0_IN_HEIGHT / 2; ++h) {
            for (int w = 0; w < CELL0_IN_WIDTH / 2; ++w) {
                float diff = std::abs(cell0_out[c][h][w] - ref_cell0_out[c][h][w]);
                if (diff > 1e-3f) { // Allowable tolerance
                    std::cout << "Mismatch at (" << c << "," << h << "," << w << "): "
                              << "model_top=" << cell0_out[c][h][w]
                              << ", ref=" << ref_cell0_out[c][h][w]
                              << ", diff=" << diff << std::endl;
                    mismatch++;
                }
            }
        }
    }

    if (mismatch == 0) {
        std::cout << "All outputs match!" << std::endl;
    } else {
        std::cout << "Mismatches found: " << mismatch << std::endl;
    }

    return 0;
}
