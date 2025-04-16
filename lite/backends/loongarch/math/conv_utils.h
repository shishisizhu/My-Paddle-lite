#pragma once
#include "lite/backends/loongarch/math/instruction_utils.h"
#include <lasxintrin.h>
#include <lsxintrin.h>
#include <vector>
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

// tranpose [chout, chin, wh, ww] to [chout/block,chin,wh,ww,block]
// dout space should be allocated before calling conv_trans_weights_numc
void conv_trans_weights_numc(const float* din,
                             float* dout,  // dout has been expanded
                             int chout,
                             int chin,
                             int wh,
                             int ww,
                             int block);

// tranpose [chout,chin,wh,ww] to [chout/block,wh,ww,chin,block]
// this function is different from conv_trans_weights_numc just
// in that we make chw->hwc
void conv_trans_weights_numc_c3(const float* din,
                                float* dout,
                                int chout,
                                int chin,
                                int wh,
                                int ww,
                                int block);

// for input and filter pack
void pack8_m256(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);
void pack4_m128(lite::Tensor* input,
                lite::Tensor* output,
                const int channel_num,
                const bool is_filter);

// for output unpack
void unpack8_m256(lite::Tensor* input, lite::Tensor* output);
void unpack4_m128(lite::Tensor* input, lite::Tensor* output);

// for input padding
void padding8_m256(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding4_m128(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int>& paddings);
void padding1_float(lite::Tensor* input,
                    lite::Tensor* output,
                    const std::vector<int>& paddings);

void pack_padding8_m256(lite::Tensor* input,
                        lite::Tensor* output,
                        const int channel_num,
                        const std::vector<int>& paddings);

// for activation - only support relu, relu6, leakyRelu, hard_swish
__m256 activation8_m256(__m256 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
__m128 activation4_m128(__m128 input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
float activation1_float(float input,
                        const lite_api::ActivationType act_type,
                        const operators::ActivationParam act_param);
void packC8_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel);

void unpackC8_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel);

template <typename Dtype>
void im2col(const Dtype* data_im,
            int channels,
            int height,
            int width,
            int kernel_h,
            int kernel_w,
            int pad_top,
            int pad_bottom,
            int pad_left,
            int pad_right,
            int stride_h,
            int stride_w,
            int dilation_h,
            int dilation_w,
            Dtype* data_col);

template <typename Dtype>
void im2col_common(const Dtype* data_im,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int pad_top,
                   int pad_bottom,
                   int pad_left,
                   int pad_right,
                   int stride_h,
                   int stride_w,
                   int dilation_h,
                   int dilation_w,
                   Dtype* data_col);

template <typename Dtype>
void im2col_s1(const Dtype* data_im,
               int channels,
               int height,
               int width,
               int kernel_h,
               int kernel_w,
               int pad_top,
               int pad_bottom,
               int pad_left,
               int pad_right,
               int dilation_h,
               int dilation_w,
               Dtype* data_col);

template <typename Dtype>
void im2col_s2(const Dtype* data_im,
               int channels,
               int height,
               int width,
               int kernel_h,
               int kernel_w,
               int pad_top,
               int pad_bottom,
               int pad_left,
               int pad_right,
               int dilation_h,
               int dilation_w,
               Dtype* data_col);

// From: https://stackoverflow.com/a/25627536
inline void transpose8_ps(__m256& row0,  // NOLINT
                          __m256& row1,  // NOLINT
                          __m256& row2,  // NOLINT
                          __m256& row3,  // NOLINT
                          __m256& row4,  // NOLINT
                          __m256& row5,  // NOLINT
                          __m256& row6,  // NOLINT
                          __m256& row7   // NOLINT
                          ) {
  __m256i __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256i __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = __lasx_xvilvl_w((__m256i)row1, (__m256i)row0);
  __t1 = __lasx_xvilvh_w((__m256i)row1, (__m256i)row0);
  __t2 = __lasx_xvilvl_w((__m256i)row3, (__m256i)row2);
  __t3 = __lasx_xvilvh_w((__m256i)row3, (__m256i)row2);
  __t4 = __lasx_xvilvl_w((__m256i)row5, (__m256i)row4);
  __t5 = __lasx_xvilvh_w((__m256i)row5, (__m256i)row4);
  __t6 = __lasx_xvilvl_w((__m256i)row7, (__m256i)row6);
  __t7 = __lasx_xvilvh_w((__m256i)row7, (__m256i)row6);
  __tt0 = lasx_m256i_shuffle_ps(__t0, __t2, lasx_mm_shuffle(1, 0, 1, 0));
  __tt1 = lasx_m256i_shuffle_ps(__t0, __t2, lasx_mm_shuffle(3, 2, 3, 2));
  __tt2 = lasx_m256i_shuffle_ps(__t1, __t3, lasx_mm_shuffle(1, 0, 1, 0));
  __tt3 = lasx_m256i_shuffle_ps(__t1, __t3, lasx_mm_shuffle(3, 2, 3, 2));
  __tt4 = lasx_m256i_shuffle_ps(__t4, __t6, lasx_mm_shuffle(1, 0, 1, 0));
  __tt5 = lasx_m256i_shuffle_ps(__t4, __t6, lasx_mm_shuffle(3, 2, 3, 2));
  __tt6 = lasx_m256i_shuffle_ps(__t5, __t7, lasx_mm_shuffle(1, 0, 1, 0));
  __tt7 = lasx_m256i_shuffle_ps(__t5, __t7, lasx_mm_shuffle(3, 2, 3, 2));
  row0 = (__m256)__lasx_xvpermi_q(__tt0, __tt4, CONVERT_IMM8(0x20));
  row1 = (__m256)__lasx_xvpermi_q(__tt1, __tt5, CONVERT_IMM8(0x20));
  row2 = (__m256)__lasx_xvpermi_q(__tt2, __tt6, CONVERT_IMM8(0x20));
  row3 = (__m256)__lasx_xvpermi_q(__tt3, __tt7, CONVERT_IMM8(0x20));
  row4 = (__m256)__lasx_xvpermi_q(__tt0, __tt4, CONVERT_IMM8(0x31));
  row5 = (__m256)__lasx_xvpermi_q(__tt1, __tt5, CONVERT_IMM8(0x31));
  row6 = (__m256)__lasx_xvpermi_q(__tt2, __tt6, CONVERT_IMM8(0x31));
  row7 = (__m256)__lasx_xvpermi_q(__tt3, __tt7, CONVERT_IMM8(0x31));
}


}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
