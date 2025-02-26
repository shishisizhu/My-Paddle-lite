#include "lite/kernels/loongarch/conv_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
//TODO
#include "lite/kernels/loongarch/conv_depthwise.h"
#include "lite/kernels/loongarch/conv_depthwise_common.h"
#include "lite/kernels/loongarch/conv_direct.h"
#include "lite/kernels/loongarch/conv_gemmlike.h"
#include "lite/kernels/loongarch/conv_winograd.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

#define PARAM_INIT                                                           \
  auto& param = this->Param<param_t>();                                      \
  auto w_dims = param.filter->dims();                                        \
  auto& ctx = this->ctx_->template As<LoongArchContext>();                   \
  auto paddings = *param.paddings;                                           \
  auto dilations = *param.dilations;                                         \
  int ic = w_dims[1] * param.groups;                                         \
  int oc = w_dims[0];                                                        \
  int kh = w_dims[2];                                                        \
  int kw = w_dims[3];                                                        \
  int pad_h = paddings[0];                                                   \
  int pad_w = paddings[2];                                                   \
  int stride = param.strides[0];                                             \
  int sh = param.strides[1];                                                 \
  int sw = param.strides[0];                                                 \
  int threads = ctx.threads();                                               \
  int chin = param.x->dims()[1];                                             \
  int hin = param.x->dims()[2];                                              \
  int win = param.x->dims()[3];                                              \
  int chout = param.output->dims()[1];                                       \
  int hout = param.output->dims()[2];                                        \
  int wout = param.output->dims()[3];                                        \
  bool pads_equal =                                                          \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));        \
  bool pads_all_equal = (pads_equal && pad_h == pad_w);                      \
  bool ks_equal = (sw == sh) && (kw == kh);                                  \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);             \
  bool kps_equal = (pad_h == pad_w) && ks_equal;                             \
  bool flag_dw_3x3 = (kw == 3) && (kh == 3) && (stride == 1 || stride == 2); \
  bool flag_dw_5x5 = (kw == 5) && (kh == 5) && (stride == 1 || stride == 2); \
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

template <>
void ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PARAM_INIT
  /// select conv impl
  if (param.groups == ic && ic == oc && ks_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1 && kw == 3 && stride == 1 && ks_equal &&
             no_dilation) {
    impl_ = new WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking winograd conv";
  } else if (param.groups == 1 && kw == 3 && stride == 2 &&
             chin * chout < 4 * hin * win && ks_equal && no_dilation) {
    impl_ = new DirectConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking direct conv";
  } else {
    impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
    // VLOG(3) << "invoking gemm like conv";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}


}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
