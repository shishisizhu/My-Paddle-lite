// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/loongarch/math/instance_norm.h"
#include <lsxintrin.h>
#include <cmath>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

void instance_norm(const float* in,
                   float* out,
                   const int n,
                   const int c,
                   const int height,
                   const int width,
                   const float epsilon,
                   const float* scale,
                   const float* bias,
                   float* saved_mean,
                   float* saved_variance) {
  int nc = n * c;
  int spatial_size = height * width;

// compute saved_mean and saved_variance
#pragma omp parallel for
  for (int i = 0; i < nc; ++i) {
    const float* in_p = in + i * spatial_size;
    float sum_spatial = 0.f;
    float summ_spatial = 0.f;
    for (int h = 0; h < height; ++h) {
      int w = width;

      __m128 sum0 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 sum1 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 sum2 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 sum3 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 square_sum0 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 square_sum1 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 square_sum2 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 square_sum3 = (__m128)__lsx_vreplgr2vr_w(0);
      __m128 in0, in1, in2, in3;
      for (; w > 15; w -= 16) {
        in0 = (__m128)__lsx_vld(in_p, 0);
        in1 = (__m128)__lsx_vld(in_p + 4, 0);
        in2 = (__m128)__lsx_vld(in_p + 8, 0);
        in3 = (__m128)__lsx_vld(in_p + 12, 0);
        // add x
        sum0 = __lsx_vfadd_s(sum0, in0);
        sum1 = __lsx_vfadd_s(sum1, in1);
        sum2 = __lsx_vfadd_s(sum2, in2);
        sum3 = __lsx_vfadd_s(sum3, in3);
        // add x * x
        square_sum0 = __lsx_vfmadd_s(in0, in0, square_sum0);
        square_sum1 = __lsx_vfmadd_s(in1, in1, square_sum1);
        square_sum2 = __lsx_vfmadd_s(in2, in2, square_sum2);
        square_sum3 = __lsx_vfmadd_s(in3, in3, square_sum3);

        in_p += 16;
      }
      for (; w > 7; w -= 8) {
        in0 = (__m128)__lsx_vld(in_p, 0);
        in1 = (__m128)__lsx_vld(in_p + 4, 0);
        sum0 = __lsx_vfadd_s(sum0, in0);
        sum1 = __lsx_vfadd_s(sum1, in1);
        square_sum0 = __lsx_vfmadd_s(in0, in0, square_sum0);
        square_sum1 = __lsx_vfmadd_s(in1, in1, square_sum1);
        in_p += 8;
      }
      for (; w > 3; w -= 4) {
        in0 = (__m128)__lsx_vld(in_p, 0);
        sum0 = __lsx_vfadd_s(sum0, in0);
        square_sum0 = __lsx_vfmadd_s(in0, in0, square_sum0);
        in_p += 4;
      }
      float sum = 0.f;
      float summ = 0.f;
      for (; w > 0; w--) {
        sum += *in_p;
        summ += (*in_p) * (*in_p);
        in_p++;
      }

      sum0 = __lsx_vfadd_s(sum0, sum1);
      sum2 = __lsx_vfadd_s(sum2, sum3);
      square_sum0 = __lsx_vfadd_s(square_sum0, square_sum1);
      square_sum2 = __lsx_vfadd_s(square_sum2, square_sum3);

      sum0 = __lsx_vfadd_s(sum0, sum2);
      square_sum0 = __lsx_vfadd_s(square_sum0, square_sum2);

      __m128 r = lsx_hadd_s(sum0, square_sum0);
      r = lsx_hadd_s(r, r);
      float buf[4];
      __lsx_vst(r, buf, 0);
      sum += buf[0];
      summ += buf[1];
      sum_spatial += sum;
      summ_spatial += summ;
    }
    float mean = sum_spatial / spatial_size;
    // float variance = summ / spatial_size - mean * mean;
    // the flolowing code has higher precision than above comment code
    float variance = (summ_spatial - mean * mean * spatial_size) / spatial_size;
    float std = 1.f / sqrtf(variance + epsilon);

    saved_mean[i] = mean;
    saved_variance[i] = std;
  }
// compute instance_norm result: out = scale * (in - mean) / std + bias
#pragma omp parallel for
  for (int i = 0; i < nc; ++i) {
    const float* in_p = in + i * spatial_size;
    float* out_p = out + i * spatial_size;
    int j = spatial_size;
    const float sstd_val =
        scale == nullptr ? saved_variance[i] : scale[i % c] * saved_variance[i];
    const float bias_val = bias == nullptr ? 0. : bias[i % c];
    const float mean_val = saved_mean[i];
    const __m128 vsstd = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&sstd_val));
    const __m128 vbias = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias_val));
    const __m128 vmean = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&mean_val));
    __m128 in0, in1, submean0, submean1, out0, out1;

    for (; j > 7; j -= 8) {
      in0 = (__m128)__lsx_vld(in_p, 0);
      in1 = (__m128)__lsx_vld(in_p + 4, 0);
      submean0 = __lsx_vfsub_s(in0, vmean);
      submean1 = __lsx_vfsub_s(in1, vmean);
      out0 = __lsx_vfmadd_s(submean0, vsstd, vbias);
      out1 = __lsx_vfmadd_s(submean1, vsstd, vbias);

      __lsx_vst(out0, out_p, 0);
      __lsx_vst(out1, out_p + 4, 0);

      in_p += 8;
      out_p += 8;
    }
    for (; j > 3; j -= 4) {
      in0 = (__m128)__lsx_vld(in_p, 0);
      submean0 = __lsx_vfsub_s(in0, vmean);
      out0 = __lsx_vfmadd_s(submean0, vsstd, vbias);

      __lsx_vst(out0, out_p, 0);

      in_p += 4;
      out_p += 4;
    }
    for (; j > 0; j--) {
      *out_p = (*in_p - mean_val) * sstd_val + bias_val;
      in_p++;
      out_p++;
    }
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
