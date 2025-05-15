#include "model.h"
#include "weights.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdint> // int16_t 타입 사용을 위해 추가



void read_input_data(const char* filename, data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH]) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "입력 파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }

    // 입력 데이터 읽기
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
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < OUT_CHANNELS; i++) {
        float val = float(output[i]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }

    std::cout << "\n모델 출력 통계:" << std::endl;
    std::cout << "최소값: " << min_val << std::endl;
    std::cout << "최대값: " << max_val << std::endl;

    float max_logit = max_val;
    float exp_values[OUT_CHANNELS];
    float sum_exp = 0.0f;

    for (int i = 0; i < OUT_CHANNELS; i++) {
        exp_values[i] = std::exp(float(output[i]) - max_logit);
        sum_exp += exp_values[i];
    }

    float max_prob = 0.0f;
    int max_class = 0;

    std::cout << "\n=== 추론 결과 ===" << std::endl;
    std::cout << "----------------" << std::endl;

    for (int i = 0; i < OUT_CHANNELS; i++) {
        float prob = exp_values[i] / sum_exp;
        std::cout << "클래스 " << i << ": " << std::fixed << std::setprecision(4) << prob * 100 << "%" << std::endl;

        if (prob > max_prob) {
            max_prob = prob;
            max_class = i;
        }
    }

    std::cout << "\n최종 예측:" << std::endl;
    std::cout << "클래스 " << max_class << " (확률: " << std::fixed << std::setprecision(4) << max_prob * 100 << "%" << std::endl;
}

int main() {
    data_t input[IN_CHANNELS][CELL0_IN_HEIGHT][CELL0_IN_WIDTH];
    data_t output[OUT_CHANNELS];
    
    initialize_weights();

    read_input_data("test.bin", input);
    model_top(input, output);
    print_inference_results(output);

    return 0;
}
