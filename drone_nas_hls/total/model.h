#ifndef MODEL_H
#define MODEL_H

// 데이터 타입 정의
typedef float data_t;  // 부동소수점 형식
typedef unsigned char bit_t; // ReLU를 위한 비트 타입

// 모델 파라미터 정의
#define IN_CHANNELS 2
#define OUT_CHANNELS 12
#define CELL0_IN_HEIGHT 32
#define CELL0_IN_WIDTH 64
#define CELL1_IN_HEIGHT 16
#define CELL1_IN_WIDTH 32

// 중간 레이어 채널 수 정의
#define STEM_CHANNELS 12
#define CELL0_CHANNELS 8
#define CELL1_CHANNELS 16


void stem_layer(
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]
);

void cell0_preprocess(
    data_t stem0_conv_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre0_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre1_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre0_op0_relu_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]
);

void cell0_layer(
    data_t pre0_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t pre1_op1_conv_output[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t concat_output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2]
);

void cell1_preprocess(
    data_t pre0_op0_relu_output[STEM_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t concat_output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2],
    data_t pre0_bn_batchnorm_output[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH],
    data_t pre1_op1_conv_output[4*CELL0_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH]
);

void cell1_layer(
    data_t pre0_bn_batchnorm_output[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH],
    data_t pre1_op1_conv_output[CELL1_CHANNELS][CELL1_IN_HEIGHT][CELL1_IN_WIDTH],
    data_t concat_output[4*CELL1_CHANNELS][CELL1_IN_HEIGHT/2][CELL1_IN_WIDTH/2]
);

void model_top(
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t output[OUT_CHANNELS]
);


#endif // MODEL_H 