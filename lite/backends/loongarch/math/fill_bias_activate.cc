/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/loongarch/math/fill_bias_activate.h"
#include <string.h>
#include <algorithm>
#include "lite/core/op_registry.h"
#include <lsxintrin.h>
#include <lasxintrin.h>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

static void activate_relu_inplace(float *data, int len, float alpha, int mode) {
  int i = 0;

  if (0 == mode) {  // relu
    __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
    for (; i + 7 < len; i += 8) {
      __m256 vec_data = (__m256)__lasx_xvld(data + i, 0);
      __lasx_xvst(__lasx_xvfmax_s(vec_data, vec_zero), data + i, 0);
    }
    __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
    for (; i + 3 < len; i += 4) {
      __m128 vec_data_128 = (__m128)__lsx_vld( data + i, 0);
      __lsx_vst(__lsx_vfmax_s(vec_data_128, vec_zero_128), data + i, 0);
    }
    for (; i < len; i++) {
      data[i] = data[i] > 0.f ? data[i] : 0.f;
    }
  } else {  // relu6
    __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));

    for (; i + 7 < len; i += 8) {
      __m256 vec_data = (__m256)__lasx_xvld(data + i, 0);
      __lasx_xvst(
          __lasx_xvfmin_s(__lasx_xvfmax_s(vec_data, vec_zero), vec_alph), data + i, 0);
    }

    __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
    __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));
    for (; i + 3 < len; i += 4) {
      __m128 vec_data_128 = (__m128)__lsx_vld( data + i, 0);
      __lsx_vst(
            __lsx_vfmin_s(__lsx_vfmax_s(vec_data_128, vec_zero_128), vec_alph_128),
            data + i,
            0);
    }

    for (; i < len; i++) {
      data[i] = data[i] > 0.f ? data[i] : 0.f;
      data[i] = data[i] < alpha ? data[i] : alpha;
    }
  }
}

static void activate_relu_inplace_bias(float *data,
                                       const float *bias,
                                       int channel,
                                       int channel_size,
                                       float alpha,
                                       int mode) {
  int i = 0;
  int j = 0;
  float *tmp_data = data;

  __m256 vec_zero = {0.f};
  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};
  __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));

  __m128 vec_zero_128 = {0.f};
  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};
  __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));


  if (0 == mode) {  // relu
    for (j = 0; j < channel; j++) {
      i = 0;
      tmp_data = data + j * channel_size;

      vec_bias = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
      for (; i + 7 < channel_size; i += 8) {
        vec_data = (__m256)__lasx_xvld(tmp_data + i, 0);
        vec_data = __lasx_xvfadd_s(vec_bias, vec_data);
        __lasx_xvst(__lasx_xvfmax_s(vec_data, vec_zero), tmp_data + i, 0);
      }

      vec_bias_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
      for (; i + 3 < channel_size; i += 4) {
        vec_data_128 = (__m128)__lsx_vld( tmp_data + i, 0);
        vec_data_128 = __lsx_vfadd_s(vec_data_128, vec_bias_128);
        __lsx_vst(__lsx_vfmax_s(vec_data_128, vec_zero_128), tmp_data + i, 0);
      }

      for (; i < channel_size; i++) {
        tmp_data[i] += bias[j];
        tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
      }
    }
  } else {  // relu6
    for (j = 0; j < channel; j++) {
      i = 0;
      tmp_data = data + j * channel_size;

      vec_bias = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
      for (; i + 7 < channel_size; i += 8) {
        vec_data = (__m256)__lasx_xvld(tmp_data + i, 0);
        vec_data = __lasx_xvfadd_s(vec_bias, vec_data);
        __lasx_xvst(__lasx_xvfmin_s(__lasx_xvfmax_s(vec_data, vec_zero), vec_alph),
            tmp_data + i, 0);
      }


      vec_bias_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
      for (; i + 3 < channel_size; i += 4) {
        vec_data_128 = (__m128)__lsx_vld( tmp_data + i, 0);
        vec_data_128 = __lsx_vfadd_s(vec_data_128, vec_bias_128);
        __lsx_vst(__lsx_vfmin_s(__lsx_vfmax_s(vec_data_128, vec_zero_128), vec_alph_128), tmp_data + i, 0);
      }

      for (; i < channel_size; i++) {
        tmp_data[i] += bias[j];
        tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : 0.f;
        tmp_data[i] = tmp_data[i] < alpha ? tmp_data[i] : alpha;
      }
    }
  }
}

static void activate_lrelu_inplace(float *data, int len, float alpha) {
  const int cmp_le_os = 2;
  int i = 0;


  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
  __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));
  for (; i + 7 < len; i += 8) {
    __m256 vec_data = (__m256)__lasx_xvld(data + i, 0);
    __m256 vec_lr = __lasx_xvfmul_s(vec_alph, vec_data);
    __m256 vec_mask = (__m256)__lasx_xvfcmp_sle_s(vec_data, vec_zero);
    __lasx_xvst(lasx_m256i_blendv_ps(vec_data, vec_lr, vec_mask), data + i, 0);
  }

  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
  __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));
  for (; i + 3 < len; i += 4) {
    __m128 vec_data_128 = (__m128)__lsx_vld( data + i, 0);
    __m128 vec_lr_128 = __lsx_vfmul_s(vec_data_128, vec_alph_128);
    __m128 vec_mask_128 = (__m128)__lsx_vfcmp_sle_s(vec_data_128, vec_zero_128);
    __lsx_vst(lsx_m128_blendv_ps(vec_data_128, vec_lr_128, vec_mask_128), data + i, 0);
  }

  for (; i < len; i++) {
    data[i] = data[i] > 0.f ? data[i] : alpha * data[i];
  }
}

static void activate_lrelu_inplace_bias(float *data,
                                        const float *bias,
                                        int channel,
                                        int channel_size,
                                        float alpha) {
  const int cmp_le_os = 2;
  int i = 0;
  int j = 0;
  float *tmp_data = data;


  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
  __m256 vec_alph = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));
  __m256 vec_bias = {0.f};

  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
  __m128 vec_alph_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&alpha));
  __m128 vec_bias_128 = {0.f};


  for (j = 0; j < channel; j++) {
    i = 0;
    tmp_data = data + j * channel_size;


    vec_bias = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
    for (; i + 7 < channel_size; i += 8) {
      __m256 vec_data = __lasx_xvfadd_s(vec_bias, (__m256)__lasx_xvld(tmp_data + i, 0));
      __m256 vec_lr = __lasx_xvfmul_s(vec_alph, vec_data);
      __m256 vec_mask = (__m256)__lasx_xvfcmp_sle_s(vec_data, vec_zero);
      __lasx_xvst(lasx_m256i_blendv_ps(vec_data, vec_lr, vec_mask), tmp_data + i, 0);
    }

    vec_bias_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
    for (; i + 3 < channel_size; i += 4) {
      __m128 vec_data_128 =
          __lsx_vfadd_s(vec_bias_128, (__m128)__lsx_vld( tmp_data + i, 0));
      __m128 vec_lr_128 = __lsx_vfmul_s(vec_data_128, vec_alph_128);
      __m128 vec_mask_128 = (__m128)__lsx_vfcmp_sle_s(vec_data_128, vec_zero_128);
      __lsx_vst(lsx_m128_blendv_ps(vec_data_128, vec_lr_128, vec_mask_128), tmp_data + i, 0);
    }

    for (; i < channel_size; i++) {
      tmp_data[i] += bias[j];
      tmp_data[i] = tmp_data[i] > 0.f ? tmp_data[i] : alpha * tmp_data[i];
    }
  }
}

static void activate_hardswish_inplace_bias(float *data,
                                            const float *bias,
                                            int channel,
                                            int channel_size,
                                            float scale,
                                            float threshold,
                                            float offset) {

  int cnt = channel_size >> 5;
  int remain = channel_size & 31;
  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
  float __scale = 1.0 / scale;
  __m256 vec_scale = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&__scale));
  __m256 vec_threshold = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&threshold));
  __m256 vec_offset = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&offset));


  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
  float _scale = 1.0 / scale;
  __m128 vec_scale_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&_scale));
  __m128 vec_threshold_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&threshold));
  __m128 vec_offset_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&offset));
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  for (int i = 0; i < channel; i++) {

    __m256 vec_bias = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias[i]));

    __m128 vec_bias_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias[i]));
    float *tmp_data = data + i * channel_size;

    for (int j = 0; j < cnt; j++) {

      __m256 vin0 = __lasx_xvfadd_s((__m256)__lasx_xvld(tmp_data, 0), vec_bias);
      __m256 vin1 = __lasx_xvfadd_s((__m256)__lasx_xvld(tmp_data + 8, 0), vec_bias);
      __m256 vin2 = __lasx_xvfadd_s((__m256)__lasx_xvld(tmp_data + 16, 0), vec_bias);
      __m256 vin3 = __lasx_xvfadd_s((__m256)__lasx_xvld(tmp_data + 24, 0), vec_bias);
      __m256 vadd0 = __lasx_xvfadd_s(vin0, vec_offset);
      __m256 vadd1 = __lasx_xvfadd_s(vin1, vec_offset);
      __m256 vadd2 = __lasx_xvfadd_s(vin2, vec_offset);
      __m256 vadd3 = __lasx_xvfadd_s(vin3, vec_offset);
      __m256 vsum0 = __lasx_xvfmul_s(vin0, vec_scale);
      __m256 vsum1 = __lasx_xvfmul_s(vin1, vec_scale);
      __m256 vsum2 = __lasx_xvfmul_s(vin2, vec_scale);
      __m256 vsum3 = __lasx_xvfmul_s(vin3, vec_scale);
      __m256 vres0 =
          __lasx_xvfmin_s(__lasx_xvfmax_s(vadd0, vec_zero), vec_threshold);
      __m256 vres1 =
          __lasx_xvfmin_s(__lasx_xvfmax_s(vadd1, vec_zero), vec_threshold);
      __m256 vres2 =
          __lasx_xvfmin_s(__lasx_xvfmax_s(vadd2, vec_zero), vec_threshold);
      __m256 vres3 =
          __lasx_xvfmin_s(__lasx_xvfmax_s(vadd3, vec_zero), vec_threshold);
      __lasx_xvst(__lasx_xvfmul_s(vres0, vsum0), tmp_data, 0);
      __lasx_xvst(__lasx_xvfmul_s(vres1, vsum1), tmp_data + 8, 0);
      __lasx_xvst(__lasx_xvfmul_s(vres2, vsum2), tmp_data + 16, 0);
      __lasx_xvst(__lasx_xvfmul_s(vres3, vsum3), tmp_data + 24, 0);
      tmp_data += 32;
    }
    for (int j = 0; j < cnt_4; j++) {
      __m128 vin0 = __lsx_vfadd_s((__m128)__lsx_vld( tmp_data, 0), vec_bias_128);
      __m128 vadd0 = __lsx_vfadd_s(vin0, vec_offset_128);
      __m128 vsum0 = __lsx_vfmul_s(vin0, vec_scale_128);
      __m128 vres0 =
          __lsx_vfmin_s(__lsx_vfmax_s(vadd0, vec_zero_128), vec_threshold_128);
      __lsx_vst(__lsx_vfmul_s(vres0, vsum0), tmp_data, 0);
      tmp_data += 4;
    }
    for (int j = 0; j < rem_4; j++) {
      tmp_data[0] = tmp_data[0] + bias[i];
      tmp_data[0] = std::min(std::max(0.f, tmp_data[0] + offset), threshold) *
                    tmp_data[0] / scale;
      tmp_data++;
    }
  }
}

static void activate_hardswish_inplace(
    float *data, int len, float scale, float threshold, float offset) {

  int cnt = len >> 5;
  int remain = len & 31;
  __m256 vec_zero = (__m256)__lasx_xvreplgr2vr_w(0);
  float __scale = 1.0/scale;
  __m256 vec_scale = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&__scale));
  __m256 vec_threshold = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&threshold));
  __m256 vec_offset = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&offset));

  __m128 vec_zero_128 = (__m128)__lsx_vreplgr2vr_w(0);
  float _scale = 1.0/scale;
  __m128 vec_scale_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&_scale));
  __m128 vec_threshold_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&threshold));
  __m128 vec_offset_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&offset));
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  float *tmp_data = data;
  for (int i = 0; i < cnt; i++) {

    __m256 vin0 = (__m256)__lasx_xvld(tmp_data, 0);
    __m256 vin1 = (__m256)__lasx_xvld(tmp_data + 8, 0);
    __m256 vin2 = (__m256)__lasx_xvld(tmp_data + 16, 0);
    __m256 vin3 = (__m256)__lasx_xvld(tmp_data + 24, 0);
    __m256 vadd0 = __lasx_xvfadd_s(vin0, vec_offset);
    __m256 vadd1 = __lasx_xvfadd_s(vin1, vec_offset);
    __m256 vadd2 = __lasx_xvfadd_s(vin2, vec_offset);
    __m256 vadd3 = __lasx_xvfadd_s(vin3, vec_offset);
    __m256 vsum0 = __lasx_xvfmul_s(vin0, vec_scale);
    __m256 vsum1 = __lasx_xvfmul_s(vin1, vec_scale);
    __m256 vsum2 = __lasx_xvfmul_s(vin2, vec_scale);
    __m256 vsum3 = __lasx_xvfmul_s(vin3, vec_scale);
    __m256 vres0 = __lasx_xvfmin_s(__lasx_xvfmax_s(vadd0, vec_zero), vec_threshold);
    __m256 vres1 = __lasx_xvfmin_s(__lasx_xvfmax_s(vadd1, vec_zero), vec_threshold);
    __m256 vres2 = __lasx_xvfmin_s(__lasx_xvfmax_s(vadd2, vec_zero), vec_threshold);
    __m256 vres3 = __lasx_xvfmin_s(__lasx_xvfmax_s(vadd3, vec_zero), vec_threshold);
    __lasx_xvst(__lasx_xvfmul_s(vres0, vsum0), tmp_data, 0);
    __lasx_xvst(__lasx_xvfmul_s(vres1, vsum1), tmp_data + 8, 0);
    __lasx_xvst(__lasx_xvfmul_s(vres2, vsum2), tmp_data + 16, 0);
    __lasx_xvst(__lasx_xvfmul_s(vres3, vsum3), tmp_data + 24, 0);
    tmp_data += 32;
  }
  for (int i = 0; i < cnt_4; i++) {
    __m128 vin0 = (__m128)__lsx_vld( tmp_data, 0);
    __m128 vadd0 = __lsx_vfadd_s(vin0, vec_offset_128);
    __m128 vsum0 = __lsx_vfmul_s(vin0, vec_scale_128);
    __m128 vres0 =
        __lsx_vfmin_s(__lsx_vfmax_s(vadd0, vec_zero_128), vec_threshold_128);
    __lsx_vst(__lsx_vfmul_s(vres0, vsum0), tmp_data, 0);
    tmp_data += 4;
  }
  for (int i = 0; i < rem_4; i++) {
    tmp_data[0] = std::min(std::max(0.f, tmp_data[0] + offset), threshold) *
                  tmp_data[0] / scale;
    tmp_data++;
  }
}

static void activate_none_inplace_bias(float *data,
                                       const float *bias,
                                       int channel,
                                       int channel_size) {
  int i = 0;
  int j = 0;
  float *tmp_data = data;

  __m256 vec_bias = {0.f};
  __m256 vec_data = {0.f};

  __m128 vec_bias_128 = {0.f};
  __m128 vec_data_128 = {0.f};


  for (j = 0; j < channel; j++) {
    i = 0;
    tmp_data = data + j * channel_size;

    vec_bias = (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
    for (; i + 7 < channel_size; i += 8) {
      vec_data = (__m256)__lasx_xvld(tmp_data + i, 0);
      vec_data = __lasx_xvfadd_s(vec_bias, vec_data);
      __lasx_xvst(vec_data, tmp_data + i, 0);
    }
    vec_bias_128 = (__m128)__lsx_vreplgr2vr_w(*reinterpret_cast<const int*>(&bias[j]));
    for (; i + 3 < channel_size; i += 4) {
      vec_data_128 = (__m128)__lsx_vld( tmp_data + i, 0);
      vec_data_128 = __lsx_vfadd_s(vec_data_128, vec_bias_128);
      __lsx_vst(vec_data_128, tmp_data + i, 0);
    }
    for (; i < channel_size; i++) {
      tmp_data[i] += bias[j];
    }
  }
}

void fill_bias_act(float *tensor,
                   const float *bias,
                   int channel,
                   int channel_size,
                   bool flag_bias,
                   const operators::ActivationParam *act_param) {
  auto act_type = act_param->active_type;
  float local_alpha = 0.f;
  int len = channel * channel_size;

  if ((act_param != nullptr) && (act_param->has_active)) {
    if ((flag_bias) && (bias != nullptr)) {
      // activate and bias
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace_bias(
            tensor, bias, channel, channel_size, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace_bias(tensor,
                                        bias,
                                        channel,
                                        channel_size,
                                        local_alpha,
                                        act_param->hard_swish_threshold,
                                        act_param->hard_swish_offset);
      }
    } else {
      // activate
      if (act_type == lite_api::ActivationType::kRelu) {
        activate_relu_inplace(tensor, len, local_alpha, 0);
      } else if (act_type == lite_api::ActivationType::kRelu6) {
        local_alpha = act_param->Relu_clipped_coef;
        activate_relu_inplace(tensor, len, local_alpha, 1);
      } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
        local_alpha = act_param->Leaky_relu_alpha;
        activate_lrelu_inplace(tensor, len, local_alpha);
      } else if (act_type == lite_api::ActivationType::kHardSwish) {
        local_alpha = act_param->hard_swish_scale;
        activate_hardswish_inplace(tensor,
                                   len,
                                   local_alpha,
                                   act_param->hard_swish_threshold,
                                   act_param->hard_swish_offset);
      }
    }
  } else {
    // only add bias
    if ((flag_bias) && (bias != nullptr))
      activate_none_inplace_bias(tensor, bias, channel, channel_size);
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
