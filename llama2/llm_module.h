#include <cmath>
#include <cstring>
#include <cstdint>
#include "config.h"

template <int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
  constexpr auto array_size = S * sizeof(float);
  float ss = 0.0f;
  float x_buff[S];
  float weight_buff[S];
  float out_buff[S];

  std::memcpy(x_buff, x, array_size);
  std::memcpy(weight_buff, weight, array_size);

  for (int j = 0; j < S; j++) {
    float x_j = x_buff[j];
    ss += x_j * x_j;
  }
  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  for (int j = 0; j < S; j++) {
    float weight_j = weight_buff[j];
    float x_j = x_buff[j];
    out_buff[j] = weight_j * (ss * x_j);
  }
  std::memcpy(o, out_buff, array_size);
}

template <int MAXSIZE>
void softmax(float *x, int size) {
  float buffer[MAXSIZE];
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    float x_i = x[i];
    if (x_i > max_val) {
      max_val = x_i;
    }
  }

  for (int i = 0; i < size; i++) {
    float x_i = expf(x[i] - max_val);
    buffer[i] = x_i;
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += buffer[i];
  }

  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; i++) {
    x[i] = buffer[i] * inv_sum;
  }
}

template <int N, int D, int GS>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];

  for (int i = 0; i < N; i++) {
    x_buffer[i] = xq[i];
  }

  for (int j = 0; j <= N - GS; j += GS) {
    xs_buffer[j / GS] = xs[j / GS];
  }

  for (int i = 0; i < D; i++) {
    float val = 0.0f;
    int8_t w_buffer[N];
    float ws_buffer[N / GS];

    const int in = i * N;
    for (int j = 0; j < N; j++) {
      w_buffer[j] = wq[j + in];
    }

    const int in_s = i * N / GS;
    const int groups = N / GS;
    for (int j = 0; j < groups; j++) {
      ws_buffer[j] = ws[in_s + j];
    }

    for (int j = 0; j <= N - GS; j += GS) {
      int32_t ival = 0;
      for (int k = 0; k < GS; k++) {
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]);
      }
      val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}
