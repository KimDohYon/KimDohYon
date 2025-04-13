#include "forward.h"
#include "config.h"
#include <cstring>
#include <cstdint>
#include <cmath>

// ==========================
// TILE CONFIGURATION
// ==========================
#define TILE_SIZE 32

// ==========================
// Tiled MatMul Definition
// ==========================
template <int N, int D>
void matmul_tiled(float xout[D], int8_t xq[N], float xs[N], int8_t wq[N * D], float ws[N * D]) {
  static constexpr int TILE = TILE_SIZE;
  float accum[D] = {0};

#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=8

  tiled_matmul:
  for (int i = 0; i < D; i += TILE) {
    for (int j = 0; j < N; j += TILE) {
      float partial[TILE] = {0};
#pragma HLS ARRAY_PARTITION variable=partial cyclic factor=8

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
// RMSNorm Implementation
// ==========================
template <int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
  constexpr auto array_size = S * sizeof(float);
  float ss = 0.0f;
  float x_buff[S];
  float weight_buff[S];
  float out_buff[S];
#pragma HLS array_partition variable = x_buff type = cyclic factor = 32
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 16
#pragma HLS array_partition variable = out_buff type = cyclic factor = 16

  std::memcpy(x_buff, x, array_size);
  std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
    float x_j = x_buff[j];
    ss += x_j * x_j;
  }
  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

norm_and_scale:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
    float weight_j = weight_buff[j];
    float x_j = x_buff[j];
    out_buff[j] = weight_j * (ss * x_j);
  }
  std::memcpy(o, out_buff, array_size);
}

// ==========================
// Softmax
// ==========================
template <int MAXSIZE>
void softmax(float *x, int size) {
  float buffer[MAXSIZE];
  float max_val = x[0];

max:
  for (int i = 1; i < size; i++) {
#pragma HLS PIPELINE
    float x_i = x[i];
    if (x_i > max_val) {
      max_val = x_i;
    }
  }

exp:
  for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
    float x_i = expf(x[i] - max_val);
    buffer[i] = x_i;
  }

  float sum = 0.0f;
sum:
  for (int i = 0; i < size; i++) {
    sum += buffer[i];
  }

  const float inv_sum = 1.0f / sum;

norm:
  for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
    x[i] = buffer[i] * inv_sum;
  }
}

// ==========================
// Main forward function
// ==========================
extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, int token, int pos, float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float *out) {
#pragma HLS INTERFACE m_axi port = transformer offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1

  auto w = &transformer->weights;
  constexpr int UNROLL_FACTOR = 16;
  static float x[config.dim];
  static float xb[config.dim];
  static float xb2[config.dim];
  static float hb[config.hidden_dim];
  static float hb2[config.hidden_dim];
  static QuantizedTensor<config.dim> xq;
  static QuantizedTensor<config.hidden_dim> hq;
  static float q[config.dim];
  static float k[(config.dim * config.n_kv_heads) / config.n_heads];
  static float v[(config.dim * config.n_kv_heads) / config.n_heads];
  static float att[config.n_heads * config.seq_len];

  constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  constexpr int kv_mul = config.n_heads / config.n_kv_heads;
  constexpr int head_size = dim / config.n_heads;

  std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

main_forward_loop:
  for (int l = 0; l < config.n_layers; l++) {
    rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);
    quantize(&xq, xb, GS);
    matmul_tiled<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
    matmul_tiled<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
    matmul_tiled<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

    for (int i = 0; i < kv_dim; i += 2) {
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      float v0_q = q[i];
      float v1_q = q[i + 1];
      q[i] = v0_q * fcr - v1_q * fci;
      q[i + 1] = v0_q * fci + v1_q * fcr;
      float v0_k = k[i];
      float v1_k = k[i + 1];
      k[i] = v0_k * fcr - v1_k * fci;
      k[i + 1] = v0_k * fci + v1_k * fcr;
    }
    for (int i = kv_dim; i < dim; i += 2) {
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      float v0 = q[i];
      float v1 = q[i + 1];
      q[i] = v0 * fcr - v1 * fci;
      q[i + 1] = v0 * fci + v1 * fcr;
    }

    int loff = l * config.seq_len * kv_dim;
    float *key_cache_row = key_cache + loff + pos * kv_dim;
    float *value_cache_row = value_cache + loff + pos * kv_dim;
    std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
    std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

    for (int h = 0; h < n_heads; h++) {
      const int q_offset = h * head_size;
      const int att_offset = h * seq_len;
      for (int t = 0; t <= pos; t++) {
#pragma HLS PIPELINE
        const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
#pragma HLS UNROLL
          score += q[i + q_offset] * key_cache[i + key_offset];
        }
        score /= sqrtf(head_size);
        att[t + att_offset] = score;
      }
      softmax<257>(att + att_offset, pos + 1);
      const int xb_offset = h * head_size;
      memset(xb + xb_offset, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
#pragma HLS PIPELINE
        const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t + att_offset];
        for (int i = 0; i < head_size; i++) {
#pragma HLS UNROLL
          xb[i + xb_offset] += a * value_cache[i + v_offset];
        }
      }
    }

    quantize(&xq, xb, GS);
    matmul_tiled<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);
    for (int i = 0; i < dim; i++) {
#pragma HLS UNROLL factor = 64 skip_exit_check
      x[i] += xb2[i];
    }

    rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
    quantize(&xq, xb, GS);
    matmul_tiled<dim, hidden_dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
    matmul_tiled<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);
    float hb_out[hidden_dim];
#pragma HLS ARRAY_PARTITION variable = hb_out type = cyclic factor = 16
    for (int i = 0; i < hidden_dim; i++) {
#pragma HLS UNROLL factor = 4
#pragma HLS PIPELINE
      float val = hb[i];
      val *= (1.0f / (1.0f + expf(-val)));
      val *= hb2[i];
      hb_out[i] = val;
    }
    std::memcpy(hb, hb_out, hidden_dim * sizeof(float));
    quantize(&hq, hb, GS);
    matmul_tiled<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);
    for (int i = 0; i < dim; i++) {
#pragma HLS UNROLL factor = 16 skip_exit_check
      x[i] += xb[i];
    }
  }

  rmsnorm<dim>(x, x, w->rms_final_weight);
  quantize(&xq, x, GS);
  matmul_tiled<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}
