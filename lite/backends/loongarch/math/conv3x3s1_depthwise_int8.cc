/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "lite/backends/loongarch/math/conv_depthwise_int8.h"
#include "lite/backends/loongarch/math/saturate.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
#define DATA_PACK(                                                          \
    vzero0, vin_00, vin_10, vzero1, vin_01, vin_11, vzero2, vin_02, vin_12) \
  __m128i va, vb, vc;                                                       \
  transpose3x4_4x4_epi(vzero0, vin_00, vin_10, va); /* 0 3 6 9 */           \
  transpose3x4_4x4_epi(vzero1, vin_01, vin_11, vb); /* 1 4 7 10 */          \
  transpose3x4_4x4_epi(vzero2, vin_02, vin_12, vc);                         \
  __lsx_vst(vzero0, doutr, 0);              \
  __lsx_vst(vzero1, doutr + 16, 0);         \
  __lsx_vst(vzero2, doutr + 32, 0);         \
  __lsx_vst(vin_00, doutr + 48, 0);         \
  __lsx_vst(vin_01, doutr + 64, 0);         \
  __lsx_vst(vin_02, doutr + 80, 0);         \
  __lsx_vst(vin_10, doutr + 96, 0);         \
  __lsx_vst(vin_11, doutr + 112, 0);        \
  __lsx_vst(vin_12, doutr + 128, 0);        \
  __lsx_vst(va, doutr + 144, 0);            \
  __lsx_vst(vb, doutr + 160, 0);            \
  __lsx_vst(vc, doutr + 176, 0);

#define RIGHT_PROCESS(dr0, dr1, dr2, doutr) \
  for (; w < win_new - 2; w++) {            \
    *doutr++ = dr0[0];                      \
    *doutr++ = dr0[1];                      \
    *doutr++ = dr0[2];                      \
    *doutr++ = 0;                           \
    *doutr++ = dr1[0];                      \
    *doutr++ = dr1[1];                      \
    *doutr++ = dr1[2];                      \
    *doutr++ = 0;                           \
    *doutr++ = dr2[0];                      \
    *doutr++ = dr2[1];                      \
    *doutr++ = dr2[2];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    dr0++;                                  \
    dr1++;                                  \
    dr2++;                                  \
  }                                         \
  if (w == win_new - 2) {                   \
    *doutr++ = dr0[0];                      \
    *doutr++ = dr0[1];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = dr1[0];                      \
    *doutr++ = dr1[1];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = dr2[0];                      \
    *doutr++ = dr2[1];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    dr0++;                                  \
    dr1++;                                  \
    dr2++;                                  \
  }                                         \
  if (w == win_new - 1) {                   \
    *doutr++ = dr0[0];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = dr1[0];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = dr2[0];                      \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    *doutr++ = 0;                           \
    dr0++;                                  \
    dr1++;                                  \
    dr2++;                                  \
  }

#define LEFT_PROCESS(dr0, dr1, dr2, doutr) \
  if (win_new >= 2) {                      \
    *doutr++ = 0;                          \
    *doutr++ = dr0[0];                     \
    *doutr++ = dr0[1];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = dr1[0];                     \
    *doutr++ = dr1[1];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = dr2[0];                     \
    *doutr++ = dr2[1];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    w++;                                   \
  } else {                                 \
    *doutr++ = 0;                          \
    *doutr++ = dr0[0];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = dr1[0];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = dr2[0];                     \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    *doutr++ = 0;                          \
    w++;                                   \
  }
#define LEFT_PROCESS_MORE(dr0, dr1, dr2, doutr) \
  for (; w < pad_w - 2; w++) {                  \
    memset(doutr, 0, sizeof(int8_t) * 16);      \
    doutr += 16;                                \
  }                                             \
  /* pad_w = 2 */                               \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = dr0[0];                            \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = dr1[0];                            \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = dr2[0];                            \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  *doutr++ = 0;                                 \
  w++;                                          \
  /* pad_w = 1 */                               \
  LEFT_PROCESS(dr0, dr1, dr2, doutr)
#define MID_PROCESS(dr0, dr1, dr2, doutr)                                    \
  for (; w < win_new - 14; w += 12) {                                        \
    __m128i vin_r0 = __lsx_vld(dr0, 0); \
    __m128i vin_r1 = __lsx_vld(dr1, 0); \
    __m128i vin_r2 = __lsx_vld(dr2, 0); \
    /* 01234567->12345678 */                                                 \
    __m128i vin_r01 = lsx_shuffle_epi8(vin_r0, vb1);                         \
    __m128i vin_r11 = lsx_shuffle_epi8(vin_r1, vb1);                         \
    __m128i vin_r21 = lsx_shuffle_epi8(vin_r2, vb1);                         \
    /* 01234567->23456789 */                                                 \
    __m128i vin_r02 = lsx_shuffle_epi8(vin_r0, vb2);                         \
    __m128i vin_r12 = lsx_shuffle_epi8(vin_r1, vb2);                         \
    __m128i vin_r22 = lsx_shuffle_epi8(vin_r2, vb2);                         \
    /* 01234567->012a 345a 678a */                                           \
    __m128i vin_00 = lsx_shuffle_epi8(vin_r0, vmask);                        \
    __m128i vin_10 = lsx_shuffle_epi8(vin_r1, vmask);                        \
    __m128i vin_20 = lsx_shuffle_epi8(vin_r2, vmask);                        \
    /* 12345678-> 123a 456a 789a */                                          \
    __m128i vin_01 = lsx_shuffle_epi8(vin_r01, vmask);                       \
    __m128i vin_11 = lsx_shuffle_epi8(vin_r11, vmask);                       \
    __m128i vin_21 = lsx_shuffle_epi8(vin_r21, vmask);                       \
    /* 23456789-> 234a 567a 8910a */                                         \
    __m128i vin_02 = lsx_shuffle_epi8(vin_r02, vmask);                       \
    __m128i vin_12 = lsx_shuffle_epi8(vin_r12, vmask);                       \
    __m128i vin_22 = lsx_shuffle_epi8(vin_r22, vmask);                       \
    /* a0b0c0d0, a1b1c1d1 -> a0a1b0b1c0d0d0d1 */                             \
    DATA_PACK(vin_00,                                                        \
              vin_10,                                                        \
              vin_20,                                                        \
              vin_01,                                                        \
              vin_11,                                                        \
              vin_21,                                                        \
              vin_02,                                                        \
              vin_12,                                                        \
              vin_22)                                                        \
    dr0 += 12;                                                               \
    dr1 += 12;                                                               \
    dr2 += 12;                                                               \
    doutr += 192;                                                            \
  }
#define MID_PROCESS_PAD_1(dr0, dr1, doutr)                                 \
  __m128i vzero0 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vzero1 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vzero2 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vin_r0 = __lsx_vld(dr0, 0); \
  __m128i vin_r1 = __lsx_vld(dr1, 0); \
  /* 01234567->12345678 */                                                 \
  __m128i vin_r01 = lsx_shuffle_epi8(vin_r0, vb1);                         \
  __m128i vin_r11 = lsx_shuffle_epi8(vin_r1, vb1);                         \
  /* 01234567->23456789 */                                                 \
  __m128i vin_r02 = lsx_shuffle_epi8(vin_r0, vb2);                         \
  __m128i vin_r12 = lsx_shuffle_epi8(vin_r1, vb2);                         \
  /* 01234567->012a 345a 678a */                                           \
  __m128i vin_00 = lsx_shuffle_epi8(vin_r0, vmask);                        \
  __m128i vin_10 = lsx_shuffle_epi8(vin_r1, vmask);                        \
  /* 12345678-> 123a 456a 789a */                                          \
  __m128i vin_01 = lsx_shuffle_epi8(vin_r01, vmask);                       \
  __m128i vin_11 = lsx_shuffle_epi8(vin_r11, vmask);                       \
  /* 23456789-> 234a 567a 8910a */                                         \
  __m128i vin_02 = lsx_shuffle_epi8(vin_r02, vmask);                       \
  __m128i vin_12 = lsx_shuffle_epi8(vin_r12, vmask);

#define TOP_MID_PAD_1                                                         \
  /* a0b0c0d0, a1b1c1d1 -> a0a1b0b1c0d0d0d1 */                                \
  DATA_PACK(                                                                  \
      vzero0, vin_00, vin_10, vzero1, vin_01, vin_11, vzero2, vin_02, vin_12) \
  dr0 += 12;                                                                  \
  dr1 += 12;                                                                  \
  doutr += 192;

#define BOT_MID_PAD_1                                                         \
  /* a0b0c0d0, a1b1c1d1 -> a0a1b0b1c0d0d0d1 */                                \
  DATA_PACK(                                                                  \
      vin_00, vin_10, vzero0, vin_01, vin_11, vzero1, vin_02, vin_12, vzero2) \
  dr0 += 12;                                                                  \
  dr1 += 12;                                                                  \
  doutr += 192;

#define MID_PROCESS_PAD_2(dr0, doutr)                                      \
  __m128i vzero0 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vzero1 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vzero2 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vin_r0 = __lsx_vld(dr0, 0); \
  __m128i vin_10 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vin_11 =  __lsx_vreplgr2vr_w(0);                                       \
  __m128i vin_12 =  __lsx_vreplgr2vr_w(0);                                       \
  /* 01234567->12345678  */                                                \
  __m128i vin_r01 = lsx_shuffle_epi8(vin_r0, vb1);                         \
  /* 01234567->23456789 */                                                 \
  __m128i vin_r02 = lsx_shuffle_epi8(vin_r0, vb2);                         \
  /* 01234567->012a 345a 678a */                                           \
  __m128i vin_00 = lsx_shuffle_epi8(vin_r0, vmask);                        \
  /* 12345678-> 123a 456a 789a */                                          \
  __m128i vin_01 = lsx_shuffle_epi8(vin_r01, vmask);                       \
  /* 23456789-> 234a 567a 8910a */                                         \
  __m128i vin_02 = lsx_shuffle_epi8(vin_r02, vmask);

#define TOP_MID_PAD_2                                                         \
  /* a0b0c0d0, a1b1c1d1 -> a0a1b0b1c0d0d0d1 */                                \
  DATA_PACK(                                                                  \
      vzero0, vin_00, vin_10, vzero1, vin_01, vin_11, vzero2, vin_02, vin_12) \
  dr0 += 12;                                                                  \
  doutr += 192;

#define BOT_MID_PAD_2                                                         \
  /* a0b0c0d0, a1b1c1d1 -> a0a1b0b1c0d0d0d1 */                                \
  DATA_PACK(                                                                  \
      vin_00, vin_10, vzero0, vin_01, vin_11, vzero1, vin_02, vin_12, vzero2) \
  dr0 += 12;                                                                  \
  doutr += 192;

// a0b0c0d0 a1b1c1d1 a2b2c2d2 -> a0a1a20 b0b1b20 c0c1c20 d0d1d20
inline void transpose3x4_4x4_epi(__m128i& row0,  // NOLINT
                                 __m128i& row1,  // NOLINT
                                 __m128i& row2,  // NOLINT
                                 __m128i& row3   // NOLINT
                                 ) {
  __m128i tmp0 = __lsx_vilvl_w(row1, row0);  // a0a1b0b1
  __m128i tmp1 = __lsx_vilvh_w(row1, row0);  // c0c1d0d1
  // int32 -> fp32
  __m128 v0 = lsx_vffint_s_w(row2);  // a2b2c2d2
  __m128 v1 = lsx_vffint_s_w(tmp0);  // a0a1b0b1
  __m128 v2 = lsx_vffint_s_w(tmp1);  // c0c1d0d1
  // a0a1a2b2
  __m128 v00 = lsx_m128i_shuffle_ps(v1, v0, lsx_mm_shuffle(0x44));
  // b0b1b2c2
  __m128 v01 = lsx_m128i_shuffle_ps(v1, v0, lsx_mm_shuffle(0x9e));  // [10, 01, 11, 10]
  // c0c1c2d2
  __m128 v02 = lsx_m128i_shuffle_ps(v2, v0, lsx_mm_shuffle(0xe4));  // [11, 10, 01, 00]
  // d0d1c2d2
  __m128 v03 = lsx_m128i_shuffle_ps(v2, v0, lsx_mm_shuffle(0xee));  // [11, 10, 11, 10]
  // fp32 -> int32
  row0 = __lsx_vftint_w_s(v00);
  row1 = __lsx_vftint_w_s(v01);
  row2 = __lsx_vftint_w_s(v02);
  row3 = __lsx_vftint_w_s(v03);
  // d0d1d2d2
  row3 = __lsx_vshuf4i_w(row3, 0xf4);  // [11, 11, 01, 00]
}

void prepack_input_im2col_s1_int8(const int8_t* din,
                                  int8_t* dout,
                                  int pad_w,
                                  int pad_h,
                                  int win,
                                  int hin,
                                  int win_round,
                                  int hin_round) {
  int h = 0;
  int8_t* dout_ptr = dout;
  const int8_t* din_ptr = din;
  int win_new = win + pad_w;
  int hin_new = hin + pad_h;
  __m128i vb1 =
      lsx_set_epi8(-127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m128i vb2 =
      lsx_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
  __m128i vmask = lsx_set_epi8(
      -127, 11, 10, 9, -127, 8, 7, 6, -127, 5, 4, 3, -127, 2, 1, 0);
  int8_t zero_ptr[32];
  memset(zero_ptr, 0, sizeof(int8_t) * 32);
  if (pad_w == 0) {
    // top
    if (pad_h == 1) {  // top only support pad_h = 0 or 1
      int w = 0;
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      int8_t* doutr = dout_ptr;
      // mid-cnt
      if (hin >= 2) {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          TOP_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      } else {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          TOP_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
      h++;
      dout_ptr += win_round;
    } else if (pad_h > 1) {
      for (; h < pad_h - 2; h++) {
        memset(dout_ptr, 0, sizeof(int8_t) * win_round);
        dout_ptr += win_round;
      }
      // pad_h = 2
      int w = 0;
      const int8_t* dr0 = din_ptr;
      int8_t* doutr = dout_ptr;
      for (; w < win_new - 14; w += 12) {
        MID_PROCESS_PAD_2(dr0, doutr)
        TOP_MID_PAD_2
      }
      if (w < win_new) {
        auto tmp_ptr = zero_ptr;
        RIGHT_PROCESS(tmp_ptr, tmp_ptr, dr0, doutr)
      }
      h++;
      dout_ptr += win_round;
      // pad_h = 1
      w = 0;
      dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      doutr = dout_ptr;
      // mid-cnt
      if (hin >= 2) {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          TOP_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      } else {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          TOP_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
      h++;
      dout_ptr += win_round;
    }
    // mid
    for (; h < hin_round && h < hin_new - 2; h++) {
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      const int8_t* dr2 = dr1 + win;
      int8_t* doutr = dout_ptr;
      int w = 0;
      MID_PROCESS(dr0, dr1, dr2, doutr)
      RIGHT_PROCESS(dr0, dr1, dr2, doutr)
      din_ptr += win;
      dout_ptr += win_round;
    }
    // bottom
    if (h < hin_round) {  // bottom
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      int8_t* doutr = dout_ptr;
      int w = 0;
      if (h == hin_new - 2) {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          BOT_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(dr0, dr1, tmp_ptr, doutr)
        }
      }
      if (h == hin_new - 1) {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          BOT_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(dr0, tmp_ptr, tmp_ptr, doutr)
        }
      }
    }
  } else if (pad_w == 1) {
    const int8_t* dr0 = din_ptr;
    const int8_t* dr1 = din_ptr + win;
    int8_t* doutr = dout_ptr;
    int w = 0;
    if (pad_h == 1) {
      auto tmp_ptr = zero_ptr;
      if (hin >= 2) {
        LEFT_PROCESS(tmp_ptr, dr0, dr1, doutr);
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          TOP_MID_PAD_1
        }
        if (w < win_new) {
          tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      } else {
        LEFT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr);
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          TOP_MID_PAD_2
        }
        if (w < win_new) {
          tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
      h++;
      dout_ptr += win_round;
    } else if (pad_h > 1) {
      for (; h < pad_h - 2; h++) {
        memset(dout_ptr, 0, sizeof(int8_t) * win_round);
        dout_ptr += win_round;
      }
      // pad_h = 2
      int w = 0;
      const int8_t* dr0 = din_ptr;
      int8_t* doutr = dout_ptr;
      auto tmp_ptr = zero_ptr;
      LEFT_PROCESS(tmp_ptr, tmp_ptr, dr0, doutr);
      for (; w < win_new - 14; w += 12) {
        MID_PROCESS_PAD_2(dr0, doutr)
        TOP_MID_PAD_2
      }
      if (w < win_new) {
        auto tmp_ptr = zero_ptr;
        RIGHT_PROCESS(tmp_ptr, tmp_ptr, dr0, doutr)
      }
      h++;
      dout_ptr += win_round;
      // pad_h = 1
      w = 0;
      dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      doutr = dout_ptr;
      tmp_ptr = zero_ptr;
      LEFT_PROCESS(tmp_ptr, dr0, dr1, doutr);
      // mid-cnt
      if (hin >= 2) {
        MID_PROCESS_PAD_1(dr0, dr1, doutr)
        TOP_MID_PAD_1
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      } else {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          TOP_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
      h++;
      dout_ptr += win_round;
    }
    // mid
    for (; h < hin_round && h < hin_new - 2; h++) {
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      const int8_t* dr2 = dr1 + win;
      int8_t* doutr = dout_ptr;
      w = 0;
      LEFT_PROCESS(dr0, dr1, dr2, doutr)
      MID_PROCESS(dr0, dr1, dr2, doutr)
      RIGHT_PROCESS(dr0, dr1, dr2, doutr)
      din_ptr += win;
      dout_ptr += win_round;
    }
    // bottom
    if (h < hin_round) {  // bottom
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      int8_t* doutr = dout_ptr;
      auto tmp_ptr0 = zero_ptr;
      int w = 0;
      if (h == hin_new - 2) {
        LEFT_PROCESS(dr0, dr1, tmp_ptr0, doutr)
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          BOT_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(dr0, dr1, tmp_ptr, doutr)
        }
      }
      if (h == hin_new - 1) {
        LEFT_PROCESS(dr0, tmp_ptr0, tmp_ptr0, doutr)
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          BOT_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(dr0, tmp_ptr, tmp_ptr, doutr)
        }
      }
    }
  } else {
    const int8_t* dr0 = din_ptr;
    const int8_t* dr1 = din_ptr + win;
    int8_t* doutr = dout_ptr;
    int w = 0;
    auto tmp_ptr = zero_ptr;
    if (pad_h == 1) {
      if (h > hin - 1) {
        LEFT_PROCESS_MORE(tmp_ptr, dr0, dr1, doutr)
      } else {
        LEFT_PROCESS_MORE(tmp_ptr, tmp_ptr, dr0, doutr)
      }
    } else if (pad_h > 1) {
      for (; h < pad_h - 2; h++) {
        memset(dout_ptr, 0, sizeof(int8_t) * win_round);
        dout_ptr += win_round;
      }
      // pad_h = 2
      int w = 0;
      const int8_t* dr0 = din_ptr;
      int8_t* doutr = dout_ptr;
      tmp_ptr = zero_ptr;
      LEFT_PROCESS_MORE(tmp_ptr, tmp_ptr, dr0, doutr);
      for (; w < win_new - 14; w += 12) {
        MID_PROCESS_PAD_2(dr0, doutr)
        TOP_MID_PAD_2
      }
      if (w < win_new) {
        tmp_ptr = zero_ptr;
        RIGHT_PROCESS(tmp_ptr, tmp_ptr, dr0, doutr)
      }
      h++;
      dout_ptr += win_round;
      // pad_h = 1
      w = 0;
      dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      doutr = dout_ptr;
      tmp_ptr = zero_ptr;
      LEFT_PROCESS_MORE(tmp_ptr, dr0, dr1, doutr);
      // mid-cnt
      if (hin >= 2) {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          TOP_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      } else {
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          TOP_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
      h++;
      dout_ptr += win_round;
    }
    // mid
    for (; h < hin_round && h < hin_new - 2; h++) {
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      const int8_t* dr2 = dr1 + win;
      int8_t* doutr = dout_ptr;
      int w = 0;
      LEFT_PROCESS_MORE(dr0, dr1, dr2, doutr)
      MID_PROCESS(dr0, dr1, dr2, doutr)
      RIGHT_PROCESS(dr0, dr1, dr2, doutr)
      din_ptr += win;
      dout_ptr += win_round;
    }
    // bottom
    if (h < hin_round) {  // bottom
      const int8_t* dr0 = din_ptr;
      const int8_t* dr1 = din_ptr + win;
      int8_t* doutr = dout_ptr;
      auto tmp_ptr0 = zero_ptr;
      w = 0;
      if (h == hin_new - 2) {
        LEFT_PROCESS_MORE(tmp_ptr0, dr1, dr0, doutr)
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_1(dr0, dr1, doutr)
          BOT_MID_PAD_1
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, dr1, doutr)
        }
      }
      if (h == hin_new - 1) {
        LEFT_PROCESS_MORE(tmp_ptr0, tmp_ptr0, dr0, doutr)
        for (; w < win_new - 14; w += 12) {
          MID_PROCESS_PAD_2(dr0, doutr)
          BOT_MID_PAD_2
        }
        if (w < win_new) {
          auto tmp_ptr = zero_ptr;
          RIGHT_PROCESS(tmp_ptr, dr0, tmp_ptr, doutr)
        }
      }
    }
  }
}
template <typename Dtype>
inline void store_data_dtype_8(Dtype* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias);

template <typename Dtype>
inline void store_data_dtype_2(Dtype* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias);

template <typename Dtype>
inline void store_data_dtype_1(Dtype* dout,
                               __m128i vin,
                               __m128 vscale,
                               __m128 vbias);

template <>
inline void store_data_dtype_8(float* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias) {
  // int32 -> fp32
  __m256 vout = __lasx_xvffint_s_w(vin);
  // * scale + bias
  __m256 vres = __lasx_xvfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  __m128 vres_0 = lasx_extracti128_lo(vres);
  __m128 vres_1 = lasx_extracti128_hi(vres);
  // a0a4b0b4
  __lsx_vst(__lsx_vilvl_w(vres_1, vres_0), dout, 0);
  // c0c4d0d4
  __lsx_vst(__lsx_vilvh_w(vres_1, vres_0), dout + 4, 0);
}
template <>
inline void store_data_dtype_8(int8_t* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias) {
  float num_neg127 = -127;
  __m128 vmax = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&num_neg127));
  // int32 -> fp32
  __m256 vout = __lasx_xvffint_s_w(vin);
  // * scale + bias
  __m256 vres = __lasx_xvfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  __m128 vres_0_0 = lasx_extracti128_lo(vres);
  __m128 vres_1_0 = lasx_extracti128_hi(vres);
  // -127
  __m128 vres_0 = lsx_m128_blendv_ps(vmax, vres_0_0, __lsx_vfcmp_clt_s(vres_0_0, vmax));
  __m128 vres_1 = lsx_m128_blendv_ps(vmax, vres_1_0, __lsx_vfcmp_clt_s(vres_1_0, vmax));
  // a0a4b0b4
  __m128 vout0 = __lsx_vilvl_w(vres_1, vres_0);
  // c0c4d0d4
  __m128 vout1 = (__m128)__lsx_vilvh_w(vres_1, vres_0);
  // fp32 -> int32
  __m128i v0_i32 = __lsx_vftint_w_s(vout0);
  __m128i v1_i32 = __lsx_vftint_w_s(vout1);
  // int32 -> int16
  __m128i v0_i16 = lsx_packs_epi32(v0_i32, v0_i32);
  __m128i v1_i16 = lsx_packs_epi32(v1_i32, v1_i32);
  // int16 -> int8
  __m128i v0_i8 = lsx_packs_epi16(v0_i16, v0_i16);
  __m128i v1_i8 = lsx_packs_epi16(v1_i16, v1_i16);
  __lsx_vstelem_d(__lsx_vilvl_w(v1_i8, v0_i8), dout, 0, 0);
}
template <>
inline void store_data_dtype_2(float* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias) {
  // int32 -> fp32
  __m256 vout = __lasx_xvffint_s_w(vin);
  // * scale + bias
  __m256 vres = __lasx_xvfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  dout[0] = (reinterpret_cast<float*>(&vres))[0];
  dout[1] = (reinterpret_cast<float*>(&vres))[4];
}
template <>
inline void store_data_dtype_2(int8_t* dout,
                               __m256i vin,
                               __m256 vscale,
                               __m256 vbias) {
  // int32 -> fp32
  __m256 vout = __lasx_xvffint_s_w(vin);
  // * scale + bias
  __m256 vres = __lasx_xvfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  float v0 = (reinterpret_cast<float*>(&vres))[0];
  float v1 = (reinterpret_cast<float*>(&vres))[4];
  v0 = v0 > -127 ? v0 : -127;
  v1 = v1 > -127 ? v1 : -127;
  dout[0] = saturate_cast<int8_t>(v0);
  dout[1] = saturate_cast<int8_t>(v1);
}
template <>
inline void store_data_dtype_1(float* dout,
                               __m128i vin,
                               __m128 vscale,
                               __m128 vbias) {
  // int32 -> fp32
  __m128 vout = lsx_vffint_s_w(vin);
  // * scale + bias
  __m128 vres = __lsx_vfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  dout[0] = (reinterpret_cast<float*>(&vres))[0];
}
template <>
inline void store_data_dtype_1(int8_t* dout,
                               __m128i vin,
                               __m128 vscale,
                               __m128 vbias) {
  // int32 -> fp32
  __m128 vout = lsx_vffint_s_w(vin);
  // * scale + bias
  __m128 vres = __lsx_vfmadd_s(vout, vscale, vbias);
  // a0b0c0d0a4b4c4d4 -> a0a4b0b4c0c4d0d4
  float v0 = (reinterpret_cast<float*>(&vres))[0];
  v0 = v0 > -127 ? v0 : -127;
  dout[0] = saturate_cast<int8_t>(v0);
}

template <typename Dtype>
void conv_3x3s1_dw_int8(Dtype* dout,
                        const int8_t* din,
                        const int8_t* weights,
                        const float* bias,
                        int num,
                        int chin,
                        int hin,
                        int win,
                        int hout,
                        int wout,
                        int pad_h,
                        int pad_w,
                        int flag_act,
                        float alpha,
                        const float* scale,
                        LoongArchContext* ctx) {
  // weights: [cout, 1, kh, kw]
  // din: [num, chin, h, w] -> [num, chin, outh, outw, 9]
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const int win_round = wout * 16;
  const int hin_round = hout + 2;

  int w_stride = 9;  // kernel_w * kernel_h;
  int omp_num = num * chin;
  int pre_in_size = hin_round * win_round;
  int cnt = wout >> 3;
  int remain = wout % 8;
  __m128i vmask = lsx_set_epi8(
      -127, -127, -127, -127, -127, 8, 7, 6, -127, 5, 4, 3, -127, 2, 1, 0);
  __m128i vone = __lsx_vreplgr2vr_h(1);
  __m256i vone_l = __lasx_xvreplgr2vr_h(1);

  int rem_cnt = remain >> 1;
  int rem_rem = remain & 1;
  bool flag_bias = bias ? true : false;
  int8_t* pre_din = static_cast<int8_t*>(
      TargetMalloc(TARGET(kLoongArch),
                   std::max(pre_in_size * omp_num * sizeof(int8_t),
                            32 * omp_num * sizeof(int8_t))));
  // LOG(INFO) << "prepack_input_im2col_s1_int8: ";
  // auto start = clock();
  for (int n = 0; n < omp_num; ++n) {
    const int8_t* din_batch = din + n * size_in_channel;
    int8_t* out_ptr = pre_din + n * pre_in_size;
    // im2col data [num, chin, h, w] -> [num, chin, outh, outw, 9 + 7(0)]
    // int8 -> int16 -> +128 ->uint16 -> int8
    prepack_input_im2col_s1_int8(
        din_batch, out_ptr, pad_w, pad_h, win, hin, win_round, hout);
  }
// auto end = clock();
// LOG(INFO) << "im2col duration: " << (end-start) * 1000.0 /CLOCKS_PER_SEC;
// start = clock();
#pragma omp parallel for
  for (int n = 0; n < omp_num; ++n) {
    int8_t* pre_din_ptr0 = pre_din + n * pre_in_size;
    Dtype* dout_batch = dout + n * size_out_channel;
    int now_c = n % chin;
    float bias_val = flag_bias ? static_cast<const float>(bias[now_c]) : 0;
    const int8_t* weight_ptr = weights + now_c * w_stride;
    __m128 vscale = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&scale[now_c]));
    __m128 vbias = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias_val));
    __m256 vscale_l = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&scale[now_c]));
    __m256 vbias_l = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias_val));
    // w00w01w02w10w11w12w20w21w22w00w01w02..
    __m128i weight_val =
        __lsx_vld(weight_ptr, 0);
    // set - w00w01w02-0-w10w11w12-0-w20w21w22-0-0000
    __m128i vw_temp = lsx_shuffle_epi8(weight_val, vmask);
    __m256i vw = lasx_set_q(vw_temp, vw_temp);
    for (int h = 0; h < hout; h++) {
      int8_t* pre_din_ptr = pre_din_ptr0;
      Dtype* dout_ptr = dout_batch;
      for (int w = 0; w < cnt; w++) {
        __m256i vin0 =
            __lasx_xvld(pre_din_ptr, 0);
        __m256i vin1 = __lasx_xvld(pre_din_ptr + 32, 0);
        __m256i vin2 = __lasx_xvld(pre_din_ptr + 64, 0);
        __m256i vin3 = __lasx_xvld(pre_din_ptr + 96, 0);
        __m256i vout0 = __lasx_xvreplgr2vr_w(0);
        __m256i vout1 = __lasx_xvreplgr2vr_w(0);
        __m256i vout2 = __lasx_xvreplgr2vr_w(0);
        __m256i vout3 = __lasx_xvreplgr2vr_w(0);

        // u8 * s8 = s16
        __m256i vsum0 = lasx_maddubs_epi16(vin0, vw);
        __m256i vsum1 = lasx_maddubs_epi16(vin1, vw);
        __m256i vsum2 = lasx_maddubs_epi16(vin2, vw);
        __m256i vsum3 = lasx_maddubs_epi16(vin3, vw);
        // s16 * s16 = s32
        vout0 = __lasx_xvmadd_h(vsum0, vone_l);
        vout1 = __lasx_xvmadd_h(vsum1, vone_l);
        vout2 = __lasx_xvmadd_h(vsum2, vone_l);
        vout3 = __lasx_xvmadd_h(vsum3, vone_l);

        // a0a2b0b2a4a6b4b6
        __m256i vres0 = lasx_hadd_epi32(vout0, vout1);
        // c0c2d0d2c4c6d4d6
        __m256i vres1 = lasx_hadd_epi32(vout2, vout3);
        // a0b0c0d0a4b4c4d4
        __m256i vres = lasx_hadd_epi32(vres0, vres1);
        store_data_dtype_8<Dtype>(dout_ptr, vres, vscale_l, vbias_l);
        dout_ptr += 8;
        pre_din_ptr += 128;
      }
      for (int w = 0; w < rem_cnt; w++) {
        __m256i vin0 =
            __lasx_xvld(pre_din_ptr, 0);
        __m256i vout0 = __lasx_xvreplgr2vr_w(0);

        // u8 * s8 = s16
        __m256i vsum0 = lasx_maddubs_epi16(vin0, vw);
        // s16 * s16 = s32
        vout0 = __lasx_xvmadd_h(vsum0, vone_l);

        // a0a2b0b2a4a6b4b6
        __m256i vres0 = lasx_hadd_epi32(vout0, vout0);
        // a0b0c0d0a4b4c4d4
        __m256i vres = lasx_hadd_epi32(vres0, vres0);
        store_data_dtype_2<Dtype>(dout_ptr, vres, vscale_l, vbias_l);
        dout_ptr += 2;
        pre_din_ptr += 32;
      }
      if (rem_rem > 0) {
        __m128i vin0 =
            __lsx_vld(pre_din_ptr, 0);
        __m128i vout0 = __lsx_vreplgr2vr_w(0);

        // u8 * s8 = s16
        __m128i vsum0 = lsx_maddubs_epi16(vin0, vw_temp);
        // s16 * s16 = s32
        vout0 = __lsx_xvmadd_h(vsum0, vone);
        // a0a2b0b2
        __m128i vres0 = lsx_hadd_w(vout0, vout0);
        // a0b0c0d0
        __m128i vres = lsx_hadd_w(vres0, vres0);
        store_data_dtype_1(dout_ptr, vres, vscale, vbias);
      }
      pre_din_ptr0 += win_round;
      dout_batch += wout;
    }
  }
  // end = clock();
  // LOG(INFO) << "compute duration: " << (end-start) * 1000.0 /CLOCKS_PER_SEC;
  TargetFree(TARGET(kLoongArch), pre_din);
}
template void conv_3x3s1_dw_int8(float* dout,
                                 const int8_t* din,
                                 const int8_t* weights,
                                 const float* bias,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 int pad_h,
                                 int pad_w,
                                 int flag_act,
                                 float alpha,
                                 const float* scale,
                                 LoongArchContext* ctx);
template void conv_3x3s1_dw_int8(int8_t* dout,
                                 const int8_t* din,
                                 const int8_t* weights,
                                 const float* bias,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 int pad_h,
                                 int pad_w,
                                 int flag_act,
                                 float alpha,
                                 const float* scale,
                                 LoongArchContext* ctx);
#undef MID_PROCESS_PAD_2
#undef TOP_MID_PAD_2
#undef BOT_MID_PAD_2
#undef MID_PROCESS_PAD_1
#undef TOP_MID_PAD_1
#undef BOT_MID_PAD_1
#undef MID_PROCESS
#undef LEFT_PROCESS_MORE
#undef LEFT_PROCESS
#undef RIGHT_PROCESS
#undef DATA_PACK
#undef ROUNDUP
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
