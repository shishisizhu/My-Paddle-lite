#pragma once

#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
void conv_depthwise_3x3s1_p01_direct(
    const float* din,
    float* dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float* weights,
    const float* bias,
    int pad,
    bool flag_bias,
    const operators::ActivationParam act_param);
void conv_depthwise_3x3s2_p01_direct(
    const float* din,
    float* dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float* weights,
    const float* bias,
    int pad,
    bool flag_bias,
    const operators::ActivationParam act_param);
void conv_depthwise_5x5s1(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param);
void conv_depthwise_5x5s2(const float* din,
                          float* dout,
                          int num,
                          int ch_out,
                          int h_out,
                          int w_out,
                          int ch_in,
                          int h_in,
                          int w_in,
                          const float* weights,
                          const float* bias,
                          int pad,
                          bool flag_bias,
                          const operators::ActivationParam act_param);
void conv_depthwise_3x3_pack(const operators::ConvParam& param,
                             lite::Tensor* input_padding_,
                             lite::Tensor* input_pack_,
                             lite::Tensor* filter_pack_,
                             lite::Tensor* output_pack_);
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
