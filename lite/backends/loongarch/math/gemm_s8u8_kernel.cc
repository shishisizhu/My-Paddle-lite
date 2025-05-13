/* Copyright (c) 2021 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "lite/backends/loongarch/math/gemm_s8u8_kernel.h"
#include "lite/backends/loongarch/math/instruction_utils.h"
#include <lasxintrin.h>
#include <lsxintrin.h>
#include <stdint.h>
#include <algorithm>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

//********************** activte and bias function **************************
void gemm_fuse_relu_bias(__m256* vec_data,
                         __m256 vec_bias,
                         __m256 vec_alph,
                         __m256 vec_zero,
                         int act_mode) {
  __m256 vec_lr, vec_mask;
  *vec_data = __lasx_xvfadd_s(*vec_data, vec_bias);
  switch (act_mode) {
    case 1:
      *vec_data = __lasx_xvfmax_s(*vec_data, vec_zero);  // relu
      break;
    case 2:
      *vec_data =
          __lasx_xvfmin_s(__lasx_xvfmax_s(*vec_data, vec_zero), vec_alph);  // relu6
      break;
    case 3:
      vec_lr = __lasx_xvfmul_s(vec_alph, *vec_data);  // lrelu
      vec_mask = (__m256)__lasx_xvfcmp_sle_s(*vec_data, vec_zero);
      *vec_data = (__m256)lasx_m256i_blendv_ps(*vec_data, vec_lr, vec_mask);
      break;
    default:
      break;
  }
}

void gemm_fuse_relu_bias_128(__m128* vec_data,
                             __m128 vec_bias,
                             __m128 vec_alph,
                             __m128 vec_zero,
                             int act_mode) {
  __m128 vec_lr_128, vec_mask_128;
  *vec_data = __lsx_vfadd_s(*vec_data, vec_bias);
  switch (act_mode) {
    case 1:
      *vec_data = __lsx_vfmax_s(*vec_data, vec_zero);
      break;
    case 2:
      *vec_data = __lsx_vfmin_s(__lsx_vfmax_s(*vec_data, vec_zero), vec_alph);
      break;
    case 3:
      vec_lr_128 = __lsx_vfmul_s(*vec_data, vec_alph);
      vec_mask_128 = (__m128)__lsx_vfcmp_sle_s(*vec_data, vec_zero);
      *vec_data = lsx_m128_blendv_ps(*vec_data, vec_lr_128, vec_mask_128);
      break;
    default:
      break;
  }
}

void gemm_fuse_relu_bias_f32(float* data,
                             float bias,
                             float alph,
                             int act_mode) {
  *data += bias;
  switch (act_mode) {
    case 1:
      *data = std::max(*data, 0.f);
      break;
    case 2:
      *data = std::min(std::max(*data, 0.f), alph);
      break;
    case 3:
      *data = *data > 0.f ? *data : alph * *data;
      break;
    default:
      break;
  }
}

#define ACT_RELU_BIAS(data, bias, mode) \
  gemm_fuse_relu_bias(&data, bias, vec_alph, vec_zero, mode);

#define ACT_RELU_BIAS_128(data, bias, mode) \
  gemm_fuse_relu_bias_128(&data, bias, vec_alph_128, vec_zero_128, mode);

#define ACT_RELU_BIAS_FP32(data, bias, mode) \
  gemm_fuse_relu_bias_f32(&data, bias, relu_alpha, mode);

//******************************** marco ************************************
#define CLIP_BORDER_LEFT (-127)
#define CLIP_BORDER_RIGHT (127)

#define CLIP_S8(a)     \
  static_cast<int8_t>( \
      std::min(std::max(a, CLIP_BORDER_LEFT), CLIP_BORDER_RIGHT))

#define FLOAT2INT(a) \
  a > 0 ? static_cast<int>(a + 0.5f) : static_cast<int>(a - 0.5f)

// extra 2 regs
#define _MM256_DOT_U8S8(dst, src1, src2, vec_tmp_marco)          \
  vec_tmp_marco = lasx_maddubs_epi16(src1, src2);              \
  vec_tmp_marco = lasx_madd_epi16(vec_tmp_marco, vec_one_s16); \
  dst = __lasx_xvadd_w(dst, vec_tmp_marco);

#define _MM_DOT_U8S8(dst, src1, src2, vec_tmp_marco)          \
  vec_tmp_marco = lsx_maddubs_epi16(src1, src2);              \
  vec_tmp_marco = lsx_madd_epi16(vec_tmp_marco, vec_one_128); \
  dst = __lsx_vadd_w(dst, vec_tmp_marco);

// 32 int to 32 int8
#define INT32x32_2_INT8x32(out, in1, in2, in3, in4)                 \
  {                                                                 \
    in1 = lasx_packs_epi32(in1, in2);                             \
    in3 = lasx_packs_epi32(in3, in4);                             \
    in4 = lasx_packs_epi16(in1, in3);                             \
    __m128i hi_in =  lasx_extracti128_hi(in4);               \
    __m128i vec_i32_2_i8_tmp =                                      \
        __lsx_vilvl_w(hi_in, lasx_extracti128_lo(in4));     \
    hi_in = __lsx_vilvh_w(hi_in, lasx_extracti128_lo(in4)); \
    out = lasx_inserti128_si256(out, vec_i32_2_i8_tmp, 0);        \
    out = lasx_inserti128_si256(out, hi_in, 1);                   \
    out = __lasx_xvmax_b(out, vec_mins_127);                       \
  }

// BroadCast K4 8-bit data to 8 lanes
#define SET_A(i, offt) \
  vec_A##i = __lasx_xvreplgr2vr_w(*reinterpret_cast<int*>(a_ptr + offt));

// BroadCast K4 8-bit data to 4 lanes
#define SET_A_128(i, offt) \
  vec_A##i##_128 = __lsx_vreplgr2vr_w(*reinterpret_cast<int*>(a_ptr + offt));

// Load K4xN8 8-bit data, total 256 bits
#define LOAD_B(i, offt) \
  vec_B##i = __lasx_xvld(b_ptr + offt, 0);

// Load K4xN4 8-bit data, total 128 bits
#define LOAD_B_128(i, offt) \
  vec_B##i##_128 =          \
      __lsx_vld(b_ptr + offt, 0);

#define SUDOT(c, b, a) _MM256_DOT_U8S8(vec_C##c, vec_B##b, vec_A##a, vec_tmp)

#define SUDOT_128(c, b, a) \
  _MM_DOT_U8S8(vec_C##c##_128, vec_B##b##_128, vec_A##a##_128, vec_tmp_128)

#define INIT_C                     \
  vec_C0 = __lasx_xvreplgr2vr_d(0); \
  vec_C1 = __lasx_xvreplgr2vr_d(0); \
  vec_C2 = __lasx_xvreplgr2vr_d(0); \
  vec_C3 = __lasx_xvreplgr2vr_d(0); \
  vec_C4 = __lasx_xvreplgr2vr_d(0); \
  vec_C5 = __lasx_xvreplgr2vr_d(0); \
  vec_C6 = __lasx_xvreplgr2vr_d(0); \
  vec_C7 = __lasx_xvreplgr2vr_d(0);

#define INIT_C_128                  \
  vec_C0_128 = __lsx_vreplgr2vr_w(0); \
  vec_C1_128 = __lsx_vreplgr2vr_w(0);

#define KERN_2x32                                                             \
  SET_A(0, 0)                                                                 \
  SET_A(1, 4)                                                                 \
  LOAD_B(0, 0)                                                                \
  LOAD_B(1, 32)                                                               \
  LOAD_B(2, 64)                                                               \
  LOAD_B(3, 96)                                                               \
  SUDOT(0, 0, 0)                                                              \
  SUDOT(1, 1, 0)                                                              \
  SUDOT(2, 2, 0)                                                              \
  SUDOT(3, 3, 0)                                                              \
  SUDOT(4, 0, 1) SUDOT(5, 1, 1) SUDOT(6, 2, 1) SUDOT(7, 3, 1) a_ptr += 2 * 4; \
  b_ptr += 32 * 4;

#define KERN_1x32                                                         \
  SET_A(0, 0)                                                             \
  LOAD_B(0, 0)                                                            \
  LOAD_B(1, 32)                                                           \
  LOAD_B(2, 64)                                                           \
  LOAD_B(3, 96)                                                           \
  SUDOT(0, 0, 0) SUDOT(1, 1, 0) SUDOT(2, 2, 0) SUDOT(3, 3, 0) a_ptr += 4; \
  b_ptr += 32 * 4;

#define KERN_2x24                                                             \
  SET_A(0, 0)                                                                 \
  SET_A(1, 4)                                                                 \
  LOAD_B(0, 0)                                                                \
  LOAD_B(1, 32)                                                               \
  LOAD_B(2, 64)                                                               \
  SUDOT(0, 0, 0)                                                              \
  SUDOT(1, 1, 0)                                                              \
  SUDOT(2, 2, 0) SUDOT(4, 0, 1) SUDOT(5, 1, 1) SUDOT(6, 2, 1) a_ptr += 2 * 4; \
  b_ptr += 24 * 4;

#define KERN_1x24                                                        \
  SET_A(0, 0)                                                            \
  LOAD_B(0, 0)                                                           \
  LOAD_B(1, 32)                                                          \
  LOAD_B(2, 64) SUDOT(0, 0, 0) SUDOT(1, 1, 0) SUDOT(2, 2, 0) a_ptr += 4; \
  b_ptr += 24 * 4;

#define KERN_2x16                                                             \
  SET_A(0, 0)                                                                 \
  SET_A(1, 4)                                                                 \
  LOAD_B(0, 0)                                                                \
  LOAD_B(1, 32)                                                               \
  SUDOT(0, 0, 0) SUDOT(1, 1, 0) SUDOT(4, 0, 1) SUDOT(5, 1, 1) a_ptr += 2 * 4; \
  b_ptr += 16 * 4;

#define KERN_1x16                                                      \
  SET_A(0, 0)                                                          \
  LOAD_B(0, 0) LOAD_B(1, 32) SUDOT(0, 0, 0) SUDOT(1, 1, 0) a_ptr += 4; \
  b_ptr += 16 * 4;

#define KERN_2x8                                                         \
  SET_A(0, 0)                                                            \
  SET_A(1, 4) LOAD_B(0, 0) SUDOT(0, 0, 0) SUDOT(4, 0, 1) a_ptr += 2 * 4; \
  b_ptr += 8 * 4;

#define KERN_1x8 \
  SET_A(0, 0)    \
  LOAD_B(0, 0)   \
  SUDOT(0, 0, 0) \
  a_ptr += 4;    \
  b_ptr += 8 * 4;

#define KERN_2x4                                                         \
  SET_A_128(0, 0)                                                        \
  SET_A_128(1, 4)                                                        \
  LOAD_B_128(0, 0) SUDOT_128(0, 0, 0) SUDOT_128(1, 0, 1) a_ptr += 2 * 4; \
  b_ptr += 4 * 4;

#define KERN_1x4     \
  SET_A_128(0, 0)    \
  LOAD_B_128(0, 0)   \
  SUDOT_128(0, 0, 0) \
  a_ptr += 4;        \
  b_ptr += 4 * 4;

#define KERN_2x2                                                         \
  SET_A_128(0, 0)                                                        \
  SET_A_128(1, 4)                                                        \
  vec_B0_128 = lsx_loadl_epi64(b_ptr); \
  SUDOT_128(0, 0, 0) SUDOT_128(1, 0, 1) a_ptr += 2 * 4;                  \
  b_ptr += 2 * 4;

#define KERN_1x2                                                         \
  SET_A_128(0, 0)                                                        \
  vec_B0_128 = lsx_loadl_epi64(b_ptr); \
  SUDOT_128(0, 0, 0)                                                     \
  a_ptr += 4;                                                            \
  b_ptr += 2 * 4;

#define STORE_32(in0, in1, in2, in3, i)                                \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]);  \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]);  \
  dst_vec_ps2 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in2), vec_scale[i]);  \
  dst_vec_ps3 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in3), vec_scale[i]);  \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                   \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                   \
  ACT_RELU_BIAS(dst_vec_ps2, vec_bias[i], relu_type)                   \
  ACT_RELU_BIAS(dst_vec_ps3, vec_bias[i], relu_type)                   \
  in0 = __lasx_xvftintrne_w_s(dst_vec_ps0);                               \
  in1 = __lasx_xvftintrne_w_s(dst_vec_ps1);                               \
  in2 = __lasx_xvftintrne_w_s(dst_vec_ps2);                               \
  in3 = __lasx_xvftintrne_w_s(dst_vec_ps3);                               \
  INT32x32_2_INT8x32(dst_vec, in0, in1, in2, in3) __lasx_xvst(dst_vec, c_ptr + i * ldc, 0);

#define STORE_24(in0, in1, in2, in3, i)                                   \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]);     \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]);     \
  dst_vec_ps2 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in2), vec_scale[i]);     \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                      \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                      \
  ACT_RELU_BIAS(dst_vec_ps2, vec_bias[i], relu_type)                      \
  in0 = __lasx_xvftintrne_w_s(dst_vec_ps0);                                  \
  in1 = __lasx_xvftintrne_w_s(dst_vec_ps1);                                  \
  in2 = __lasx_xvftintrne_w_s(dst_vec_ps2);                                  \
  INT32x32_2_INT8x32(dst_vec, in0, in1, in2, in3) lasx_maskstore_epi32(c_ptr + i * ldc, vec_mask, dst_vec);

#define STORE_16(in0, in1, in2, in3, i)                               \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                  \
  in0 = __lasx_xvftintrne_w_s(dst_vec_ps0);                              \
  in1 = __lasx_xvftintrne_w_s(dst_vec_ps1);                              \
  INT32x32_2_INT8x32(dst_vec, in0, in1, in2, in3) dst_vec_128 =       \
      lasx_extracti128_lo(dst_vec);                                \
  __lsx_vst(dst_vec_128, c_ptr + i * ldc, 0);

#define STORE_8(in0, in1, in2, in3, i)                                \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  in0 = __lasx_xvftintrne_w_s(dst_vec_ps0);                              \
  INT32x32_2_INT8x32(dst_vec, in0, in1, in2, in3) dst_vec_128 =       \
      lasx_extracti128_lo(dst_vec);                                \
  __lsx_vstelm_d(dst_vec_128, c_ptr + i * ldc, 0, 0);

// __m128
#define STORE_4(in0, i)                                                   \
  {                                                                       \
    dst_vec_ps0_128 = __lsx_vfmul_s(__lsx_vffint_s_w(in0), vec_scale_128[i]); \
    ACT_RELU_BIAS_128(dst_vec_ps0_128, vec_bias_128[i], relu_type)        \
    in0 = __lsx_vftint_w_s(dst_vec_ps0_128);                               \
    in0 = __lsx_vmin_w(__lsx_vmax_w(in0, vec_left), vec_right);         \
    int* ptr = reinterpret_cast<int*>(&in0);                              \
    *(c_ptr + i * ldc) = static_cast<int8_t>(ptr[0]);                     \
    *(c_ptr + i * ldc + 1) = static_cast<int8_t>(ptr[1]);                 \
    *(c_ptr + i * ldc + 2) = static_cast<int8_t>(ptr[2]);                 \
    *(c_ptr + i * ldc + 3) = static_cast<int8_t>(ptr[3]);                 \
  }

#define STORE_2(in0, i)                                      \
  {                                                          \
    int* in0_ptr = reinterpret_cast<int*>(&in0);             \
    float bias_data = (*(bias_ptr + idx_m + i));             \
    float in0_f32 = in0_ptr[0] * (*(scale_ptr + idx_m + i)); \
    ACT_RELU_BIAS_FP32(in0_f32, bias_data, relu_type)        \
    int in0_int = FLOAT2INT(in0_f32);                        \
    *(c_ptr + i * ldc) = CLIP_S8(in0_int);                   \
    in0_f32 = in0_ptr[1] * (*(scale_ptr + idx_m + i));       \
    ACT_RELU_BIAS_FP32(in0_f32, bias_data, relu_type)        \
    in0_int = FLOAT2INT(in0_f32);                            \
    *(c_ptr + i * ldc + 1) = CLIP_S8(in0_int);               \
  }

void gemm_kernel_loop_int8(int M,
                           int N,
                           int K,
                           int8_t* A,
                           uint8_t* B,
                           int8_t* C,
                           int ldc,
                           const float* scale,
                           const float* bias,
                           int relu_type,
                           float relu_alpha) {
  int8_t* a_ptr = A;
  int8_t* c_ptr = C;
  uint8_t* b_ptr = B;
  const float* scale_ptr = scale;
  const float* bias_ptr = bias;
  int k_loop = (K + 3) >> 2;
  int pack_k = k_loop << 2;
  int idx_n = 0, idx_m = 0, idx_k = 0;

  // total 16 regs
  __m256i vec_C0, vec_C1, vec_C2, vec_C3;
  __m256i vec_C4, vec_C5, vec_C6, vec_C7;
  __m256i vec_B0, vec_B1, vec_B2, vec_B3;
  __m256i vec_A0, vec_A1, vec_tmp;
  __m256i vec_one_s16 = __lasx_xvreplgr2vr_h(static_cast<int16_t>(1));
  // save result
  __m256i dst_vec;
  __m256 vec_bias[2];
  __m256 vec_scale[2];
  __m256 dst_vec_ps0, dst_vec_ps1, dst_vec_ps2, dst_vec_ps3;
  // bias and relu
  __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&relu_alpha));
  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
  // val is in -127, 127, the other side using packs to guarantee
  __m256i vec_mins_127 = __lasx_xvreplgr2vr_b(static_cast<char>(CLIP_BORDER_LEFT));

  // SSE
  __m128i vec_C0_128, vec_C1_128;
  __m128i vec_B0_128;
  __m128i vec_A0_128, vec_A1_128, vec_tmp_128;
  __m128i vec_one_128 = __lsx_vreplgr2vr_h(static_cast<int16_t>(1));
  // save result
  __m128i dst_vec_128;
  __m128 vec_bias_128[2];
  __m128 vec_scale_128[2];
  __m128 dst_vec_ps0_128;
  // bias and relu
  __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&relu_alpha));
  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
  // clip
  __m128i vec_left = __lsx_vreplgr2vr_w(static_cast<int>(CLIP_BORDER_LEFT));
  __m128i vec_right = __lsx_vreplgr2vr_w(static_cast<int>(CLIP_BORDER_RIGHT));

  // mask load, store
  int mask0[8] = {-1, -1, -1, -1, -1, -1, 0, 0};  // load or save 24 int8-data
  __m256i vec_mask =
      __lasx_xvld(mask0, 0);

  // block A
  for (idx_m = 0; idx_m + 1 < M; idx_m += 2) {
    c_ptr = C;
    b_ptr = B;
    a_ptr = A;
    C += 2 * ldc;

    // bias and scale
    vec_bias[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_bias[1] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m + 1));
    vec_scale[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_scale[1] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m + 1));
    vec_bias_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_bias_128[1] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m + 1));
    vec_scale_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_scale_128[1] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m + 1));

    // block B
    for (idx_n = 0; idx_n + 31 < N; idx_n += 32) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x32
      }
      STORE_32(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      STORE_32(vec_C4, vec_C5, vec_C6, vec_C7, 1)
      c_ptr += 32;
    }
    for (; idx_n + 23 < N; idx_n += 24) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x24
      }
      STORE_24(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      STORE_24(vec_C4, vec_C5, vec_C6, vec_C7, 1)
      c_ptr += 24;
    }
    for (; idx_n + 15 < N; idx_n += 16) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x16
      }
      STORE_16(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      STORE_16(vec_C4, vec_C5, vec_C6, vec_C7, 1)
      c_ptr += 16;
    }
    for (; idx_n + 7 < N; idx_n += 8) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x8
      }
      STORE_8(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      STORE_8(vec_C4, vec_C5, vec_C6, vec_C7, 1)
      c_ptr += 8;
    }
    for (; idx_n + 3 < N; idx_n += 4) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x4
      }
      STORE_4(vec_C0_128, 0)
      STORE_4(vec_C1_128, 1)
      c_ptr += 4;
    }
    for (; idx_n + 1 < N; idx_n += 2) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x2
      }
      STORE_2(vec_C0_128, 0)
      STORE_2(vec_C1_128, 1)
      c_ptr += 2;
    }
    for (; idx_n < N; idx_n++) {
      a_ptr = A;
      float acc0 = 0;
      float acc1 = 0;
      float bias0 = (*(bias_ptr + idx_m));
      float bias1 = (*(bias_ptr + idx_m + 1));
      float scale0 = (*(scale_ptr + idx_m));
      float scale1 = (*(scale_ptr + idx_m + 1));
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        for (int k = 0; k < 4; k++) {
          acc0 +=
              static_cast<int>(a_ptr[k]) * static_cast<int>(b_ptr[k]) * scale0;
          acc1 += static_cast<int>(a_ptr[k + 4]) * static_cast<int>(b_ptr[k]) *
                  scale1;
        }
        a_ptr += 2 * 4;
        b_ptr += 4;
      }
      ACT_RELU_BIAS_FP32(acc0, bias0, relu_type)
      ACT_RELU_BIAS_FP32(acc1, bias1, relu_type)
      int iacc0 = FLOAT2INT(acc0);
      int iacc1 = FLOAT2INT(acc1);
      int8_t acc0_s8 = CLIP_S8(iacc0);
      int8_t acc1_s8 = CLIP_S8(iacc1);
      c_ptr[0] = acc0_s8;
      c_ptr[ldc] = acc1_s8;
      c_ptr++;
    }
    A += 2 * pack_k;
  }
  for (; idx_m < M; idx_m += 1) {
    c_ptr = C;
    b_ptr = B;
    a_ptr = A;
    C += ldc;

    // bias and scale
    vec_bias[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_scale[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_bias_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_scale_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));

    // block B
    for (idx_n = 0; idx_n + 31 < N; idx_n += 32) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x32
      }
      STORE_32(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      c_ptr += 32;
    }
    for (; idx_n + 23 < N; idx_n += 24) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x24
      }
      STORE_24(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      c_ptr += 24;
    }
    for (; idx_n + 15 < N; idx_n += 16) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x16
      }
      STORE_16(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      c_ptr += 16;
    }
    for (; idx_n + 7 < N; idx_n += 8) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x8
      }
      STORE_8(vec_C0, vec_C1, vec_C2, vec_C3, 0)
      c_ptr += 8;
    }
    for (; idx_n + 3 < N; idx_n += 4) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x4
      }
      STORE_4(vec_C0_128, 0)
      c_ptr += 4;
    }
    for (; idx_n + 1 < N; idx_n += 2) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x2
      }
      STORE_2(vec_C0_128, 0)
      c_ptr += 2;
    }
    for (; idx_n < N; idx_n++) {
      a_ptr = A;
      float acc0 = 0;
      float bias0 = (*(bias_ptr + idx_m));
      float scale0 = (*(scale_ptr + idx_m));
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        for (int k = 0; k < 4; k++) {
          acc0 +=
              static_cast<int>(a_ptr[k]) * static_cast<int>(b_ptr[k]) * scale0;
        }
        a_ptr += 4;
        b_ptr += 4;
      }
      ACT_RELU_BIAS_FP32(acc0, bias0, relu_type)
      int iacc0 = FLOAT2INT(acc0);
      int8_t acc0_s8 = CLIP_S8(iacc0);
      c_ptr[0] = acc0_s8;
      c_ptr++;
    }
    A += pack_k;
  }
}

#define STORE_32_float(in0, in1, in2, in3, i)                         \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]); \
  dst_vec_ps2 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in2), vec_scale[i]); \
  dst_vec_ps3 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in3), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps2, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps3, vec_bias[i], relu_type)                  \
  __lasx_xvst(dst_vec_ps0, c_ptr + i * ldc, 0);                     \
  __lasx_xvst(dst_vec_ps1, c_ptr + i * ldc + 8, 0);                 \
  __lasx_xvst(dst_vec_ps2, c_ptr + i * ldc + 16, 0);                \
  __lasx_xvst(dst_vec_ps3, c_ptr + i * ldc + 24, 0);

#define STORE_24_float(in0, in1, in2, in3, i)                         \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]); \
  dst_vec_ps2 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in2), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps2, vec_bias[i], relu_type)                  \
  __lasx_xvst(dst_vec_ps0, c_ptr + i * ldc, 0);                     \
  __lasx_xvst(dst_vec_ps1, c_ptr + i * ldc + 8, 0);                 \
  __lasx_xvst(dst_vec_ps2, c_ptr + i * ldc + 16, 0);

#define STORE_16_float(in0, in1, in2, in3, i)                         \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  dst_vec_ps1 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in1), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  ACT_RELU_BIAS(dst_vec_ps1, vec_bias[i], relu_type)                  \
  __lasx_xvst(dst_vec_ps0, c_ptr + i * ldc, 0);                     \
  __lasx_xvst(dst_vec_ps1, c_ptr + i * ldc + 8, 0);

#define STORE_8_float(in0, in1, in2, in3, i)                          \
  dst_vec_ps0 = __lasx_xvfmul_s(__lasx_xvffint_s_w(in0), vec_scale[i]); \
  ACT_RELU_BIAS(dst_vec_ps0, vec_bias[i], relu_type)                  \
  __lasx_xvst(dst_vec_ps0, c_ptr + i * ldc, 0);

// __m128
#define STORE_4_float(in0, i)                                             \
  {                                                                       \
    dst_vec_ps0_128 = __lsx_vfmul_s(__lsx_vffint_s_w(in0), vec_scale_128[i]); \
    ACT_RELU_BIAS_128(dst_vec_ps0_128, vec_bias_128[i], relu_type)        \
    __lsx_vst(dst_vec_ps0_128, c_ptr + i * ldc, 0);                      \
  }

#define STORE_2_float(in0, i)                                \
  {                                                          \
    int* in0_ptr = reinterpret_cast<int*>(&in0);             \
    float bias_data = (*(bias_ptr + idx_m + i));             \
    float in0_f32 = in0_ptr[0] * (*(scale_ptr + idx_m + i)); \
    ACT_RELU_BIAS_FP32(in0_f32, bias_data, relu_type)        \
    *(c_ptr + i * ldc) = in0_f32;                            \
    in0_f32 = in0_ptr[1] * (*(scale_ptr + idx_m + i));       \
    ACT_RELU_BIAS_FP32(in0_f32, bias_data, relu_type)        \
    *(c_ptr + i * ldc + 1) = in0_f32;                        \
  }

void gemm_kernel_loop_int8(int M,
                           int N,
                           int K,
                           int8_t* A,
                           uint8_t* B,
                           float* C,
                           int ldc,
                           const float* scale,
                           const float* bias,
                           int relu_type,
                           float relu_alpha) {
  int8_t* a_ptr = A;
  float* c_ptr = C;
  uint8_t* b_ptr = B;
  const float* scale_ptr = scale;
  const float* bias_ptr = bias;
  int k_loop = (K + 3) >> 2;
  int pack_k = k_loop << 2;
  int idx_n = 0, idx_m = 0, idx_k = 0;

  // total 16 regs
  __m256i vec_C0, vec_C1, vec_C2, vec_C3;
  __m256i vec_C4, vec_C5, vec_C6, vec_C7;
  __m256i vec_B0, vec_B1, vec_B2, vec_B3;
  __m256i vec_A0, vec_A1, vec_tmp;
  __m256i vec_one_s16 = __lasx_xvreplgr2vr_h(static_cast<int16_t>(1));
  // save result
  __m256 vec_bias[2];
  __m256 vec_scale[2];
  __m256 dst_vec_ps0, dst_vec_ps1, dst_vec_ps2, dst_vec_ps3;
  // bias and relu
  __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&relu_alpha));
  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);

  // SSE
  __m128i vec_C0_128, vec_C1_128;
  __m128i vec_B0_128;
  __m128i vec_A0_128, vec_A1_128, vec_tmp_128;
  __m128i vec_one_128 = __lsx_vreplgr2vr_h(static_cast<int16_t>(1));
  // save result
  __m128 vec_bias_128[2];
  __m128 vec_scale_128[2];
  __m128 dst_vec_ps0_128;
  // bias and relu
  __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&relu_alpha));
  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);

  // block A
  for (idx_m = 0; idx_m + 1 < M; idx_m += 2) {
    c_ptr = C;
    b_ptr = B;
    a_ptr = A;
    C += 2 * ldc;

    // bias and scale
    vec_bias[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_bias[1] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m + 1));
    vec_scale[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_scale[1] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m + 1));
    vec_bias_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_bias_128[1] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m + 1));
    vec_scale_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_scale_128[1] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m + 1));

    // block B
    for (idx_n = 0; idx_n + 31 < N; idx_n += 32) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x32
      }
      STORE_32_float(vec_C0, vec_C1, vec_C2, vec_C3, 0)
          STORE_32_float(vec_C4, vec_C5, vec_C6, vec_C7, 1) c_ptr += 32;
    }
    for (; idx_n + 23 < N; idx_n += 24) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x24
      }
      STORE_24_float(vec_C0, vec_C1, vec_C2, vec_C3, 0)
          STORE_24_float(vec_C4, vec_C5, vec_C6, vec_C7, 1) c_ptr += 24;
    }
    for (; idx_n + 15 < N; idx_n += 16) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x16
      }
      STORE_16_float(vec_C0, vec_C1, vec_C2, vec_C3, 0)
          STORE_16_float(vec_C4, vec_C5, vec_C6, vec_C7, 1) c_ptr += 16;
    }
    for (; idx_n + 7 < N; idx_n += 8) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x8
      }
      STORE_8_float(vec_C0, vec_C1, vec_C2, vec_C3, 0)
          STORE_8_float(vec_C4, vec_C5, vec_C6, vec_C7, 1) c_ptr += 8;
    }
    for (; idx_n + 3 < N; idx_n += 4) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x4
      }
      STORE_4_float(vec_C0_128, 0) STORE_4_float(vec_C1_128, 1) c_ptr += 4;
    }
    for (; idx_n + 1 < N; idx_n += 2) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_2x2
      }
      STORE_2_float(vec_C0_128, 0) STORE_2_float(vec_C1_128, 1) c_ptr += 2;
    }
    for (; idx_n < N; idx_n++) {
      a_ptr = A;
      float acc0 = 0;
      float acc1 = 0;
      float bias0 = (*(bias_ptr + idx_m));
      float bias1 = (*(bias_ptr + idx_m + 1));
      float scale0 = (*(scale_ptr + idx_m));
      float scale1 = (*(scale_ptr + idx_m + 1));
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        for (int k = 0; k < 4; k++) {
          acc0 +=
              static_cast<int>(a_ptr[k]) * static_cast<int>(b_ptr[k]) * scale0;
          acc1 += static_cast<int>(a_ptr[k + 4]) * static_cast<int>(b_ptr[k]) *
                  scale1;
        }
        a_ptr += 2 * 4;
        b_ptr += 4;
      }
      ACT_RELU_BIAS_FP32(acc0, bias0, relu_type)
      ACT_RELU_BIAS_FP32(acc1, bias1, relu_type)
      c_ptr[0] = acc0;
      c_ptr[ldc] = acc1;
      c_ptr++;
    }
    A += 2 * pack_k;
  }
  for (; idx_m < M; idx_m += 1) {
    c_ptr = C;
    b_ptr = B;
    a_ptr = A;
    C += ldc;

    // bias and scale
    vec_bias[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_scale[0] = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));
    vec_bias_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(bias_ptr + idx_m));
    vec_scale_128[0] = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(scale_ptr + idx_m));

    // block B
    for (idx_n = 0; idx_n + 31 < N; idx_n += 32) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x32
      }
      STORE_32_float(vec_C0, vec_C1, vec_C2, vec_C3, 0) c_ptr += 32;
    }
    for (; idx_n + 23 < N; idx_n += 24) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x24
      }
      STORE_24_float(vec_C0, vec_C1, vec_C2, vec_C3, 0) c_ptr += 24;
    }
    for (; idx_n + 15 < N; idx_n += 16) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x16
      }
      STORE_16_float(vec_C0, vec_C1, vec_C2, vec_C3, 0) c_ptr += 16;
    }
    for (; idx_n + 7 < N; idx_n += 8) {
      a_ptr = A;
      INIT_C
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x8
      }
      STORE_8_float(vec_C0, vec_C1, vec_C2, vec_C3, 0) c_ptr += 8;
    }
    for (; idx_n + 3 < N; idx_n += 4) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x4
      }
      STORE_4_float(vec_C0_128, 0) c_ptr += 4;
    }
    for (; idx_n + 1 < N; idx_n += 2) {
      a_ptr = A;
      INIT_C_128
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        KERN_1x2
      }
      STORE_2_float(vec_C0_128, 0) c_ptr += 2;
    }
    for (; idx_n < N; idx_n++) {
      a_ptr = A;
      float acc0 = 0;
      float bias0 = (*(bias_ptr + idx_m));
      float scale0 = (*(scale_ptr + idx_m));
      for (idx_k = 0; idx_k < k_loop; idx_k++) {
        for (int k = 0; k < 4; k++) {
          acc0 +=
              static_cast<int>(a_ptr[k]) * static_cast<int>(b_ptr[k]) * scale0;
        }
        a_ptr += 4;
        b_ptr += 4;
      }
      ACT_RELU_BIAS_FP32(acc0, bias0, relu_type)
      c_ptr[0] = acc0;
      c_ptr++;
    }
    A += pack_k;
  }
}

#undef ACT_RELU_BIAS
#undef ACT_RELU_BIAS_128
#undef ACT_RELU_BIAS_FP32
#undef CLIP_BORDER_LEFT
#undef CLIP_BORDER_RIGHT
#undef CLIP_S8
#undef FLOAT2INT
#undef _MM256_DOT_U8S8
#undef _MM_DOT_U8S8
#undef INT32x32_2_INT8x32
#undef SET_A
#undef SET_A_128
#undef LOAD_B
#undef LOAD_B_128
#undef SUDOT
#undef SUDOT_128
#undef INIT_C
#undef INIT_C_128
#undef KERN_2x32
#undef KERN_1x32
#undef KERN_2x24
#undef KERN_1x24
#undef KERN_2x16
#undef KERN_1x16
#undef KERN_2x8
#undef KERN_1x8
#undef KERN_2x4
#undef KERN_1x4
#undef KERN_2x2
#undef KERN_1x2
#undef STORE_32
#undef STORE_24
#undef STORE_16
#undef STORE_8
#undef STORE_4
#undef STORE_2
#undef STORE_32_float
#undef STORE_24_float
#undef STORE_16_float
#undef STORE_8_float
#undef STORE_4_float
#undef STORE_2_float

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle

