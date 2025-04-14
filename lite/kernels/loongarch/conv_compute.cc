#include "lite/kernels/loongarch/conv_compute.h"
#include <utility>
#include "lite/backends/x86/math/fill_bias_activate.h"
#include "lite/kernels/loongarch/conv_depthwise.h"
//TODO
//#include "lite/kernels/x86/conv_direct.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

#define INIT_PARAM                      \
  auto& param = this->Param<param_t>(); \
  auto x_dims = param.x->dims();        \
  auto w_dims = param.filter->dims();   \
  auto o_dims = param.output->dims();   \
  int win = x_dims[3];                  \
  int hin = x_dims[2];                  \
  int chin = x_dims[1];                 \
  int num = x_dims[0];                  \
  int wout = o_dims[3];                 \
  int hout = o_dims[2];                 \
  int chout = o_dims[1];                \
  int kw = w_dims[3];                   \
  int kh = w_dims[2];                   \
  int group = param.groups;             \
  int m = chout / group;                \
  int n = hout * wout;                  \
  int k = chin * kw * kh / group;

#define PREPARE_PARAM                                                         \
  auto& param = this->Param<param_t>();                                       \
  const int input_channel = param.x->dims()[1];                               \
  const int output_channel = param.filter->dims()[0];                         \
  const int groups = param.groups;                                            \
  const int kernel_h = param.filter->dims()[2];                               \
  const int kernel_w = param.filter->dims()[3];                               \
  const int stride_h = param.strides[0];                                      \
  const int stride_w = param.strides[1];                                      \
  auto paddings = *param.paddings;                                            \
  auto dilations = *param.dilations;                                          \
  bool dw_kernel = (input_channel == groups && output_channel == groups);     \
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);           \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);              \
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;                  \
  bool pads_equal =                                                           \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));         \
  bool flag_dw_3x3 =                                                          \
      (kernel_h == 3) && (kernel_w == 3) && (stride_h == 1 || stride_h == 2); \
  bool flag_dw_5x5 =                                                          \
      (kernel_h == 5) && (kernel_w == 5) && (stride_h == 1 || stride_h == 2);

#define PREPARE_PARAM_INT8                                          \
  auto& param = this->Param<param_t>();                             \
  const int input_channel = param.x->dims()[1];                     \
  const int output_channel = param.filter->dims()[0];               \
  const int groups = param.groups;                                  \
  const int kernel_h = param.filter->dims()[2];                     \
  const int kernel_w = param.filter->dims()[3];                     \
  const int stride_h = param.strides[0];                            \
  const int stride_w = param.strides[1];                            \
  auto paddings = *param.paddings;                                  \
  auto dilations = *param.dilations;                                \
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w); \
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;        \
  bool pads_equal =                                                 \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PREPARE_PARAM
  //add conv_5x5_depthwise implement
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;//depthwise gemm
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  bool nodilations = true;
  for (auto ele : *(param.dilations))
    if (ele != 1) nodilations = false;

  bool pad_all_equal = (paddings[0] == paddings[1]) &&
                       (paddings[1] == paddings[2]) &&
                       (paddings[2] == paddings[3]);
  bool flag_p = paddings[0] <= stride_h;


  //! select conv impl
  if (dw_kernel && kps_equal && flag_dw && pads_equal &&
      ((flag_dw_5x5 && no_dilation) || (flag_dw_3x3 && (groups & 3) == 0))) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    VLOG(3) << "invoking conv_depthwise_3x3p0p1 or conv_depthwise_5x5";
  }

/*
  // support 3x3s1p01,5x5s1p01,7x7s1p01
  //  3x3s2p012,5x5s1p012,7x7s1p012
  if (output_channel % 8 == 0 && groups == 1 &&
      (kernel_h == 3 || kernel_h == 5 || kernel_h == 7) &&
      (stride_h == 2 || stride_h == 1) && nodilations && kps_equal &&
      pad_all_equal && flag_p) {
//TODO
    impl_ = new DirectConv<PRECISION(kFloat), PRECISION(kFloat)>();
    VLOG(3) << "invoking directConv";
  }
*/

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  if (impl_) {
    return impl_->Run();
  }
  auto& ctx = ctx_->As<LoongArchContext>();
  INIT_PARAM
  bool flag_bias = (param.bias != nullptr);
  unsigned int group_size_out = m * n;
  unsigned int group_size_weights = m * k;
  unsigned int group_size_coldata = n * k;
  unsigned int channel_in_size = chin * hin * win;
  unsigned int channel_out_size = chout * hout * wout;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  float* col_data = nullptr;

  if (!flag_1x1gemm_) {
    size_t col_size = group_size_coldata * group;
    size_t col_data_size = static_cast<size_t>(col_size * sizeof(float));
    col_data = static_cast<float*>(TargetMalloc(TARGET(kLoongArch), col_data_size));
  }
  auto act_param = param.activation_param;
  paddle::lite::loongarch::math::Blas<lite::TargetType::kLoongArch> matmul(ctx);
  for (int i = 0; i < num; i++) {
    const float* din_batch = din + i * channel_in_size;
    float* dout_batch = dout + i * channel_out_size;
    const float* din_data = din_batch;
    if (!flag_1x1gemm_) {
      lite::loongarch::math::im2col<float>(din_batch,
                                     chin,
                                     hin,
                                     win,
                                     w_dims[2],
                                     w_dims[3],
                                     paddings[0],
                                     paddings[1],
                                     paddings[2],
                                     paddings[3],
                                     param.strides[0],
                                     param.strides[1],
                                     dilations[0],
                                     dilations[1],
                                     col_data);
      din_data = static_cast<const float*>(col_data);
    }

    for (int g = 0; g < group; g++) {
      const float* col_data_group = din_data + g * group_size_coldata;
      const float* weights_group = weights + g * group_size_weights;
      float* dout_group = dout_batch + g * group_size_out;
      if (n == 1) {
        matmul.GEMV<float>(
            false, m, k, 1.f, weights_group, col_data_group, 0.f, dout_group);
      } else {
        matmul.GEMM<float>(false,
                           false,
                           m,
                           n,
                           k,
                           1.f,
                           weights_group,
                           k,
                           col_data_group,
                           n,
                           0.f,
                           dout_group,
                           n);
      }
    }
    //! bias and activate
    lite::x86::math::fill_bias_act(
        dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
  }
  if (!flag_1x1gemm_) TargetFree(TARGET(kX86), col_data);
}


#undef PREPARE_PARAM
#undef PREPARE_PARAM_INT8
#undef INIT_PARAM
}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

