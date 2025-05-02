#include "shape_config.h"
#include "layers.h"
#include <hls_stream.h>
#include <cstdio>

void run_inference(
    short *in,
    float *out,
    float conv_weights[CONV_OUT_CH][1][KERNEL_H][KERNEL_W],
    float conv_bias[CONV_OUT_CH],
    float fc1_weights[FC1_OUT][FC1_IN], float fc1_bias[FC1_OUT],
    float fc2_weights[FC2_OUT][FC2_IN], float fc2_bias[FC2_OUT],
    float fc3_weights[FC3_OUT][FC3_IN], float fc3_bias[FC3_OUT],
    float fc4_weights[FC4_OUT][FC4_IN], float fc4_bias[FC4_OUT]) {

#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0 depth=IN_SIZE
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=OUT_SIZE
#pragma HLS INTERFACE m_axi port=conv_weights bundle=gmem2 depth=1024
#pragma HLS INTERFACE m_axi port=conv_bias    bundle=gmem2 depth=16
#pragma HLS INTERFACE m_axi port=fc1_weights  bundle=gmem3 depth=16384
#pragma HLS INTERFACE m_axi port=fc1_bias     bundle=gmem3 depth=128
#pragma HLS INTERFACE m_axi port=fc2_weights  bundle=gmem4 depth=8192
#pragma HLS INTERFACE m_axi port=fc2_bias     bundle=gmem4 depth=64
#pragma HLS INTERFACE m_axi port=fc3_weights  bundle=gmem5 depth=2048
#pragma HLS INTERFACE m_axi port=fc3_bias     bundle=gmem5 depth=32
#pragma HLS INTERFACE m_axi port=fc4_weights  bundle=gmem6 depth=384
#pragma HLS INTERFACE m_axi port=fc4_bias     bundle=gmem6 depth=12

#pragma HLS INTERFACE s_axilite port=in           bundle=control
#pragma HLS INTERFACE s_axilite port=out          bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control
#pragma HLS INTERFACE s_axilite port=conv_weights bundle=control
#pragma HLS INTERFACE s_axilite port=conv_bias    bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_bias     bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_bias     bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_bias     bundle=control
#pragma HLS INTERFACE s_axilite port=fc4_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=fc4_bias     bundle=control

    float input_norm[INPUT_H][INPUT_W];
    float pool_out[CONV_OUT_CH][POOL_OUT_H][POOL_OUT_W];
    float flat_out[FLAT_SIZE];
    float fc1_out[FC1_OUT], fc2_out[FC2_OUT], fc3_out[FC3_OUT], fc4_out[FC4_OUT];

    // 입력 정규화
    for (int i = 0; i < INPUT_H; ++i)
        for (int j = 0; j < INPUT_W; ++j)
            input_norm[i][j] = ((float)in[i * INPUT_W + j]) / 32768.0f * 1000;

    // 스트림 선언
    hls::stream<float> conv_relu_stream("conv_relu_stream");
    hls::stream<float> relu_pool_stream("relu_pool_stream");

    // 스트림 기반 연산
    conv2d_stream(input_norm, conv_relu_stream, conv_weights, conv_bias);
    relu_stream(conv_relu_stream, relu_pool_stream, CONV_OUT_CH * CONV_OUT_H);
    maxpool_stream(relu_pool_stream, pool_out);
    flatten_module(pool_out, flat_out);

    // FC 계층
    fc1_module(flat_out, fc1_out, fc1_weights, fc1_bias);
    relu_fc<FC1_OUT>(fc1_out);
    fc2_module(fc1_out, fc2_out, fc2_weights, fc2_bias);
    relu_fc<FC2_OUT>(fc2_out);
    fc3_module(fc2_out, fc3_out, fc3_weights, fc3_bias);
    relu_fc<FC3_OUT>(fc3_out);
    fc4_module(fc3_out, fc4_out, fc4_weights, fc4_bias);

    // 출력 저장
    for (int i = 0; i < FC4_OUT; ++i)
        out[i] = fc4_out[i];
}