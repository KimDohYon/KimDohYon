#pragma once

#define INPUT_H       2048
#define INPUT_W       2
#define CONV_OUT_CH   16
#define KERNEL_H      32
#define KERNEL_W      2
#define STRIDE_H      1
#define STRIDE_W      1

#define CONV_OUT_H    2017
#define CONV_OUT_W    1

#define POOL_KH       16
#define POOL_KW       1
#define POOL_OUT_H    126
#define POOL_OUT_W    1

#define FLAT_SIZE     2016

#define FC1_IN        FLAT_SIZE
#define FC1_OUT       128
#define FC2_IN        FC1_OUT
#define FC2_OUT       64
#define FC3_IN        FC2_OUT
#define FC3_OUT       32
#define FC4_IN        FC3_OUT
#define FC4_OUT       12

#define IN_SIZE       (INPUT_H * INPUT_W)
#define OUT_SIZE      FC4_OUT