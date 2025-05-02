#include "shape_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <string>

// ✅ inference 함수 선언 (만약 layers.h에 없다면 추가)
extern void run_inference(
    short *in,
    float *out,
    float conv_weights[CONV_OUT_CH][1][KERNEL_H][KERNEL_W],
    float conv_bias[CONV_OUT_CH],
    float fc1_weights[FC1_OUT][FC1_IN],
    float fc1_bias[FC1_OUT],
    float fc2_weights[FC2_OUT][FC2_IN],
    float fc2_bias[FC2_OUT],
    float fc3_weights[FC3_OUT][FC3_IN],
    float fc3_bias[FC3_OUT],
    float fc4_weights[FC4_OUT][FC4_IN],
    float fc4_bias[FC4_OUT]
);

// --------- 하이퍼파라미터 ---------
#define INPUT_FILE "test.bin"

// --------- 버퍼 선언 ---------
short input_data[INPUT_H * INPUT_W];
float output[FC4_OUT] = {0};
float prob[FC4_OUT] = {0};

float conv_weights[CONV_OUT_CH][1][KERNEL_H][KERNEL_W];
float conv_bias[CONV_OUT_CH];
float fc1_weights[FC1_OUT][FC1_IN];
float fc1_bias[FC1_OUT];
float fc2_weights[FC2_OUT][FC2_IN];
float fc2_bias[FC2_OUT];
float fc3_weights[FC3_OUT][FC3_IN];
float fc3_bias[FC3_OUT];
float fc4_weights[FC4_OUT][FC4_IN];
float fc4_bias[FC4_OUT];

// --------- 텍스트 로드 함수 ---------
template<typename T>
bool load_txt(const std::string& filename, T* array, size_t count) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "❌ 파일 열기 실패: " << filename << std::endl;
        return false;
    }

    size_t idx = 0;
    T value;
    while (infile >> value && idx < count) {
        array[idx++] = value;
    }

    infile.close();
    if (idx != count) {
        std::cerr << "❌ 불러온 값 수 부족 (" << idx << " / " << count << ")\n";
        return false;
    }

    return true;
}


// --------- 이진 입력 로드 함수 ---------
bool load_input_bin(const std::string &filename, short *array, size_t count) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "❌ 입력 파일 열기 실패: " << filename << std::endl;
        return false;
    }
    infile.read(reinterpret_cast<char *>(array), count * sizeof(short));
    infile.close();
    return true;
}

// --------- 소프트맥스 ---------
void softmax(const float in[FC4_OUT], float out[FC4_OUT]) {
    float max_val = in[0];
    for (int i = 1; i < FC4_OUT; ++i)
        if (in[i] > max_val) max_val = in[i];

    float sum = 0.0f;
    for (int i = 0; i < FC4_OUT; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < FC4_OUT; ++i)
        out[i] /= sum;
}

// --------- 메인 ---------
int main() {
    std::cout << "📥 입력 로드 중...\n";
    if (!load_input_bin(INPUT_FILE, input_data, INPUT_H * INPUT_W)) return 1;

    std::cout << "📥 가중치 로드 중...\n";
    if (!load_txt("fv.0.weight.txt", &conv_weights[0][0][0][0], CONV_OUT_CH * 1 * KERNEL_H * KERNEL_W)) return 1;
    if (!load_txt("fv.0.bias.txt", conv_bias, CONV_OUT_CH)) return 1;

    if (!load_txt("classifier.0.weight.txt", &fc1_weights[0][0], FC1_OUT * FC1_IN)) return 1;
    if (!load_txt("classifier.0.bias.txt", fc1_bias, FC1_OUT)) return 1;

    if (!load_txt("classifier.2.weight.txt", &fc2_weights[0][0], FC2_OUT * FC2_IN)) return 1;
    if (!load_txt("classifier.2.bias.txt", fc2_bias, FC2_OUT)) return 1;

    if (!load_txt("classifier.4.weight.txt", &fc3_weights[0][0], FC3_OUT * FC3_IN)) return 1;
    if (!load_txt("classifier.4.bias.txt", fc3_bias, FC3_OUT)) return 1;

    if (!load_txt("classifier.6.weight.txt", &fc4_weights[0][0], FC4_OUT * FC4_IN)) return 1;
    if (!load_txt("classifier.6.bias.txt", fc4_bias, FC4_OUT)) return 1;

    std::cout << "🚀 추론 시작\n";
    run_inference(  // ✅ 이름 변경된 함수 호출
        input_data, output,
        conv_weights, conv_bias,
        fc1_weights, fc1_bias,
        fc2_weights, fc2_bias,
        fc3_weights, fc3_bias,
        fc4_weights, fc4_bias
    );

    softmax(output, prob);

    std::cout << "\n=== 🔢 Raw Logits ===\n";
    for (int i = 0; i < FC4_OUT; ++i)
        std::cout << "Class " << i << ": " << output[i] << "\n";

    std::cout << "\n=== 📊 Softmax ===\n";
    for (int i = 0; i < FC4_OUT; ++i)
        std::cout << "Class " << i << ": " << prob[i] << "\n";

    int predicted = 0;
    float max_prob = prob[0];
    for (int i = 1; i < FC4_OUT; ++i) {
        if (prob[i] > max_prob) {
            max_prob = prob[i];
            predicted = i;
        }
    }

    std::cout << "\n🎯 예측 클래스: " << predicted << " (신뢰도: " << max_prob << ")\n";
    return 0;
}
