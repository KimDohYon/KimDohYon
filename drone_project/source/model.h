#ifndef MODEL_H
#define MODEL_H

// Data type definitions
typedef float data_t;          // Floating-point data type
typedef unsigned char bit_t;   // Bit type (used for ReLU masking or flags)

// Model parameter definitions
#define IN_CHANNELS 2
#define OUT_CHANNELS 12

#define CELL0_IN_HEIGHT 32
#define CELL0_IN_WIDTH 64
#define CELL1_IN_HEIGHT 16
#define CELL1_IN_WIDTH 32

#define STEM_CHANNELS 12
#define CELL0_CHANNELS 8
#define CELL1_CHANNELS 16

// Top-level model function declaration
void model_top(
    data_t inputA[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],     // First input feature map
    data_t inputB[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],     // Second input feature map
    data_t output[4 * CELL0_CHANNELS][CELL0_IN_HEIGHT / 2][CELL0_IN_WIDTH / 2], // Final output feature map

    // Convolution weights and biases (already padded to CELL0_CHANNELS)
    data_t conv_weight0[CELL0_CHANNELS][1][5][5], data_t* conv_bias0,
    data_t conv_weight1[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias1,
    data_t conv_weight2[CELL0_CHANNELS][1][3][3], data_t* conv_bias2,
    data_t conv_weight3[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias3,
    data_t conv_weight4[CELL0_CHANNELS][1][5][5], data_t* conv_bias4,
    data_t conv_weight5[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias5,
    data_t conv_weight6[CELL0_CHANNELS][1][5][5], data_t* conv_bias6,
    data_t conv_weight7[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias7,
    data_t conv_weight8[CELL0_CHANNELS][1][5][5], data_t* conv_bias8,
    data_t conv_weight9[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias9,
    data_t conv_weight10[CELL0_CHANNELS][1][5][5], data_t* conv_bias10,
    data_t conv_weight11[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias11,
    data_t conv_weight12[CELL0_CHANNELS][1][5][5], data_t* conv_bias12,
    data_t conv_weight13[CELL0_CHANNELS][CELL0_CHANNELS][1][1], data_t* conv_bias13
);

#endif // MODEL_H
