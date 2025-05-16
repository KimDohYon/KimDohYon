#ifndef MODEL_H
#define MODEL_H

// 데이터 타입 정의
typedef float data_t;          // 부동소수점 형식
typedef unsigned char bit_t;   // ReLU를 위한 비트 타입

// 모델 파라미터 정의
#define IN_CHANNELS 2
#define OUT_CHANNELS 12

#define CELL0_IN_HEIGHT 32
#define CELL0_IN_WIDTH 64
#define CELL1_IN_HEIGHT 16
#define CELL1_IN_WIDTH 32

#define STEM_CHANNELS 12
#define CELL0_CHANNELS 8
#define CELL1_CHANNELS 16

#endif // MODEL_H
