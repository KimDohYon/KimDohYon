#include <cstdint>
#include <cstring>

// ==========================
// TILE CONFIGURATION
// ==========================
#define TILE_SIZE 128

// ==========================
// Tiled MatMul Definition
// ==========================
template <int N, int D>
void matmul_tiled(float xout[D], int8_t xq[N], float xs[N], int8_t wq[N * D], float ws[N * D]) {
  static constexpr int TILE = TILE_SIZE;

  // temporary buffers
  float accum[D] = {0};

  tiled_matmul:
  for (int i = 0; i < D; i += TILE) {
    for (int j = 0; j < N; j += TILE) {
      float partial[TILE] = {0};

      for (int ti = 0; ti < TILE; ti++) {
        float val = 0.0f;
        for (int tj = 0; tj < TILE; tj++) {
          int x_idx = j + tj;
          int w_idx = (i + ti) * N + x_idx;
          int8_t x_val = xq[x_idx];
          int8_t w_val = wq[w_idx];
          float x_scale = xs[x_idx];
          float w_scale = ws[w_idx];
          val += (float)(x_val * w_val) * x_scale * w_scale;
        }
        partial[ti] = val;
      }

      for (int ti = 0; ti < TILE; ti++) {
        accum[i + ti] += partial[ti];
      }
    }
  }

  for (int i = 0; i < D; i++) {
    xout[i] = accum[i];
  }
}

// ==========================
// Top wrapper for matmul<768,768> using tile
// ==========================
extern "C" {
void top_matmul_tiled_768_768(int8_t xq[768], float xs[768],
                               int8_t wq[768 * 768], float ws[768 * 768],
                               float out[768]) {
  matmul_tiled<768, 768>(out, xq, xs, wq, ws);
}
}
