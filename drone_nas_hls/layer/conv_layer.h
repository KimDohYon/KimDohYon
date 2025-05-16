#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "model.h"

template<int IN_CH, int OUT_CH, int IN_H, int IN_W, int OUT_H, int OUT_W, int K, int GROUP>
void conv_layer(
    data_t input[IN_CH][IN_H][IN_W],
    data_t weight[OUT_CH][IN_CH / GROUP][K][K],
    data_t* bias,
    data_t output[OUT_CH][OUT_H][OUT_W],
    int stride,
    int padding,
    int group,
    int dilation
);

template<int IN_CH, int OUT_CH, int IN_H, int IN_W, int OUT_H, int OUT_W, int K, int GROUP>
void conv_layer(
    data_t input[IN_CH][IN_H][IN_W],
    data_t weight[OUT_CH][IN_CH / GROUP][K][K],
    data_t* bias,
    data_t output[OUT_CH][OUT_H][OUT_W],
    int stride,
    int padding,
    int group,
    int dilation
) {
    const int in_ch_per_group = IN_CH / group;
    const int out_ch_per_group = OUT_CH / group;

    for (int g = 0; g < group; g++) {
        for (int oc = 0; oc < out_ch_per_group; oc++) {
            const int oc_global = g * out_ch_per_group + oc;

            for (int oh = 0; oh < OUT_H; oh++) {
                for (int ow = 0; ow < OUT_W; ow++) {
                    data_t sum = (bias != nullptr) ? bias[oc_global] : 0;

                    for (int ic = 0; ic < in_ch_per_group; ic++) {
                        const int ic_global = g * in_ch_per_group + ic;

                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                int ih = oh * stride - padding + kh * dilation;
                                int iw = ow * stride - padding + kw * dilation;

                                if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                    sum += input[ic_global][ih][iw] * weight[oc_global][ic][kh][kw];
                                }
                            }
                        }
                    }

                    output[oc_global][oh][ow] = sum;
                }
            }
        }
    }
}


#endif // CONV_LAYER_H