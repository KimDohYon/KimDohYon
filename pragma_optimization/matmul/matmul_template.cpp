#include <cmath>
#include <cstdint>
#include <cstring>

static constexpr int GS = 64;  // Quantization group size

template <int N, int D>
void matmul(float xout[D], int8_t xq[N], float xs[N / GS], int8_t wq[D * N], float ws[D * (N / GS)]) {
  // xq: quantized input vector of shape (N,)
  // xs: scaling factor per input group of shape (N/GS,)
  // wq: quantized weight matrix of shape (D, N) in row-major
  // ws: scaling factor per row group in wq: shape (D, N/GS)
  // xout: result vector of shape (D,)

  int8_t x_buffer[N];
  float xs_buffer[N / GS];

  // <<<PRAGMA_x_buff_partition>>>
#pragma HLS array_partition variable = x_buffer type = cyclic factor = __X_BUFF_PARTITION__

  // <<<PRAGMA_xs_buff_partition>>>
#pragma HLS array_partition variable = xs_buffer type = cyclic factor = __XS_BUFF_PARTITION__

x_copy:
  for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
    x_buffer[i] = xq[i];
  }

xs_copy:
  for (int j = 0; j < N / GS; j++) {
#pragma HLS PIPELINE II=1
    xs_buffer[j] = xs[j];
  }

out_loop:
  for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE

    int8_t w_buffer[N];
    float ws_buffer[N / GS];

    // <<<PRAGMA_w_buff_partition>>>
#pragma HLS array_partition variable = w_buffer type = cyclic factor = __W_BUFF_PARTITION__

    // <<<PRAGMA_ws_buff_partition>>>
#pragma HLS array_partition variable = ws_buffer type = cyclic factor = __WS_BUFF_PARTITION__

  wq_copy:
    for (int j = 0; j < N; j++) {
      w_buffer[j] = wq[i * N + j];
    }

  ws_copy:
    for (int j = 0; j < N / GS; j++) {
      ws_buffer[j] = ws[i * (N / GS) + j];
    }

    float acc = 0.0f;

  group_loop:
    for (int j = 0; j < N; j += GS) {
      int32_t sum = 0;

    dot_loop:
      for (int k = 0; k < GS; k++) {
        sum += (int32_t)x_buffer[j + k] * (int32_t)w_buffer[j + k];
      }

      acc += ((float)sum) * xs_buffer[j / GS] * ws_buffer[j / GS];
    }

    xout[i] = acc;
  }
}

extern "C" {
void top_matmul(float o[2048], int8_t xq[768], float xs[12],
                    int8_t wq[2048 * 768], float ws[2048 * 12]) {

  matmul<768, 2048>(o, xq, xs, wq, ws);
}
}


