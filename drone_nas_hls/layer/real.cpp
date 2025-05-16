#include "model.h"
#include "cell_layer.h"

// === cell1 weight를 cell0 크기로 패딩하는 유틸리티 함수 ===
void pad_conv_weights(
    data_t dst[CELL0_CHANNELS][CELL0_CHANNELS][1][1],
    const data_t src[CELL1_CHANNELS][CELL1_CHANNELS][1][1]
) {
    for (int oc = 0; oc < CELL0_CHANNELS; ++oc) {
        for (int ic = 0; ic < CELL0_CHANNELS; ++ic) {
            if (oc < CELL1_CHANNELS && ic < CELL1_CHANNELS) {
                dst[oc][ic][0][0] = src[oc][ic][0][0];
            } else {
                dst[oc][ic][0][0] = 0.0f;
            }
        }
    }
}

void model_top(
    data_t inputA[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t inputB[CELL0_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH],
    data_t output[4*CELL0_CHANNELS][CELL0_IN_HEIGHT/2][CELL0_IN_WIDTH/2],

    // Conv weights and biases (이미 CELL0_CHANNELS 크기로 패딩되어 있음)
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
) {
#pragma HLS interface mode=ap_ctrl_hs port=return
#pragma HLS interface mode=m_axi bundle=gmem port=inputA offset=slave
#pragma HLS interface mode=m_axi bundle=gmem port=inputB offset=slave
#pragma HLS interface mode=m_axi bundle=gmem port=output offset=slave
// (필요 시 각 weight들도 m_axi에 붙일 수 있음)

    cell_layer<CELL0_CHANNELS, CELL0_IN_HEIGHT, CELL0_IN_WIDTH>(
        inputA, inputB, output,
        conv_weight0, conv_bias0,
        conv_weight1, conv_bias1,
        conv_weight2, conv_bias2,
        conv_weight3, conv_bias3,
        conv_weight4, conv_bias4,
        conv_weight5, conv_bias5,
        conv_weight6, conv_bias6,
        conv_weight7, conv_bias7,
        conv_weight8, conv_bias8,
        conv_weight9, conv_bias9,
        conv_weight10, conv_bias10,
        conv_weight11, conv_bias11,
        conv_weight12, conv_bias12,
        conv_weight13, conv_bias13
    );
}

