#include <cmath>
#include <cstring>

#define S 768

void rmsnorm(float o[S], float x[S], float weight[S])
{
  constexpr auto array_size = S * sizeof(float);
  float ss = 0.0f;
  float x_buff[S];
  float weight_buff[S];
  float out_buff[S];

// <<<PRAGMA_x_buff_partition>>>
#pragma HLS array_partition variable = x_buff type = cyclic factor = __X_BUFF_PARTITION__

// <<<PRAGMA_weight_buff_partition>>>
#pragma HLS array_partition variable = weight_buff type = cyclic factor = __WEIGHT_BUFF_PARTITION__

// <<<PRAGMA_out_buff_partition>>>
#pragma HLS array_partition variable = out_buff type = cyclic factor = __OUT_BUFF_PARTITION__

  std::memcpy(x_buff, x, array_size);
  std::memcpy(weight_buff, weight, array_size);

// <<<LOOP_sum_of_squares>>>
sum_of_squares:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = __SUM_UNROLL__ skip_exit_check
    float x_j = x_buff[j];
    ss += x_j * x_j;
  }

  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

// <<<LOOP_norm_and_scale>>>
norm_and_scale:
  for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = __NORM_UNROLL__
    float weight_j = weight_buff[j];
    float x_j = x_buff[j];
    out_buff[j] = weight_j * (ss * x_j);
  }

  std::memcpy(o, out_buff, array_size);
}
