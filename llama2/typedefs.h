#include <stdint.h>
#include <stdio.h>

#ifndef TYPEDEFS
#define TYPEDEFS

struct Config {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;
  int GS;
};

template <int SIZE>
struct QuantizedTensor {
  int8_t q[SIZE];
  float s[SIZE];
};

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights {
  QuantizedTensor<vocab_size * dim> q_tokens[1];
  float token_embedding_table[vocab_size * dim];

  float rms_att_weight[n_layers * dim];
  float rms_ffn_weight[n_layers * dim];

  QuantizedTensor<dim *(dim / n_heads) * n_heads> wq[n_layers];
  QuantizedTensor<dim *(dim / n_heads) * n_kv_heads> wk[n_layers];
  QuantizedTensor<dim *(dim / n_heads) * n_kv_heads> wv[n_layers];
  QuantizedTensor<n_heads * dim *(dim / n_heads)> wo[n_layers];

  QuantizedTensor<dim * hidden_dim> w1[n_layers];
  QuantizedTensor<dim * hidden_dim> w2[n_layers];
  QuantizedTensor<dim * hidden_dim> w3[n_layers];
  float rms_final_weight[dim];
  QuantizedTensor<vocab_size * dim> wcls[1];
};

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer {
  Config config;
  TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights;
};

#endif