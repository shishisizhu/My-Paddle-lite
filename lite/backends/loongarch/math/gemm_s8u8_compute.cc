#include "lite/backends/loongarch/math/gemm_s8u8_compute.h"
#include <cmath>
namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template <>
void generate_gemm_s8u8_loongarch_kern<int8_t>::repack_bias(bool is_trans,
                                                      int M,
                                                      int K,
                                                      const float *bias,
                                                      float *out,
                                                      float *Sa,
                                                      float Sb,
                                                      float Sc,
                                                      const int8_t *A) {
  const int8_t *a_ptr = A;
  for (int i = 0; i < M; i++) {
    float bias_val = bias ? bias[i] : 0.f;
    float sum = 0.f;
    float scale = (Sa[i] * Sb) * TRANS_INT8_UINT8_OFFT;
    a_ptr = A + i * K;
    if (is_trans) {
      for (int j = 0; j < K; j++) {
        sum += A[i + j * M] * scale;
      }
    } else {
      for (int j = 0; j < K; j++) {
        sum += a_ptr[j] * scale;
      }
    }
    out[i] = bias_val - sum;
    out[i] = out[i] / Sc;
  }
}

template <>
void generate_gemm_s8u8_loongarch_kern<float>::repack_bias(bool is_trans,
                                                     int M,
                                                     int K,
                                                     const float *bias,
                                                     float *out,
                                                     float *Sa,
                                                     float Sb,
                                                     float Sc,
                                                     const int8_t *A) {
  const int8_t *a_ptr = A;
  for (int i = 0; i < M; i++) {
    float bias_val = bias ? bias[i] : 0.f;
    float sum = 0.f;
    float scale = (Sa[i] * Sb) * TRANS_INT8_UINT8_OFFT;
    a_ptr = A + i * K;
    if (is_trans) {
      for (int j = 0; j < K; j++) {
        sum += A[i + j * M] * scale;
      }
    } else {
      for (int j = 0; j < K; j++) {
        sum += a_ptr[j] * scale;
      }
    }
    out[i] = bias_val - sum;
  }
}

template <>
void generate_gemm_s8u8_loongarch_kern<int8_t>::calc_scale(
    int M, float *Sa, float Sb, float Sc, float *out) {
  for (int i = 0; i < M; i++) {
    out[i] = (Sa[i] * Sb) / Sc;
  }
}

template <>
void generate_gemm_s8u8_loongarch_kern<float>::calc_scale(
    int M, float *Sa, float Sb, float Sc, float *out) {
  for (int i = 0; i < M; i++) {
    out[i] = (Sa[i] * Sb);
  }
}

template <>
void generate_gemm_s8u8_loongarch_kern<int8_t>::calc_block(
    int M, int N, int K, int *blk_m, int *blk_n) {
  int block_size, scale_tmp;
  int block_m, block_n;

  block_m = M;
  block_n = 32 * _unroll_n;
  // C(int8) + A(int8) + B(int8) + runtime packB(uint8)
  block_size = block_m * block_n + _k_align4 * (block_m + 2 * block_n);
  scale_tmp = static_cast<int>(ceil(block_size * 1.f / _l2_size));
  scale_tmp = (scale_tmp + 1) / 2;
  scale_tmp = scale_tmp * 2;
  block_n = block_n / scale_tmp;
  block_n = block_n / _unroll_n;
  block_n = block_n * _unroll_n;
  block_n = std::max(block_n, _unroll_n);

  *blk_m = block_m;
  *blk_n = block_n;
}

template <>
void generate_gemm_s8u8_loongarch_kern<float>::calc_block(
    int M, int N, int K, int *blk_m, int *blk_n) {
  int block_size, scale_tmp;
  int block_m, block_n;

  block_m = M;
  block_n = 32 * _unroll_n;
  // C(int8) + A(int8) + B(int8) + runtime packB(uint8)
  block_size =
      block_m * block_n * sizeof(float) + _k_align4 * (block_m + 2 * block_n);
  scale_tmp = static_cast<int>(ceil(block_size * 1.f / _l2_size));
  scale_tmp = (scale_tmp + 1) / 2;
  scale_tmp = scale_tmp * 2;
  block_n = block_n / scale_tmp;
  block_n = block_n / _unroll_n;
  block_n = block_n * _unroll_n;
  block_n = std::max(block_n, _unroll_n);

  *blk_m = block_m;
  *blk_n = block_n;
}

} // namespace math
} // namespace loongarch
} // namespace lite
} // namespace paddle
