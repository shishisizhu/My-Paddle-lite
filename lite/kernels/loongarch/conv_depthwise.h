#pragma once

#include <string>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

template <PrecisionType Ptype, PrecisionType OutType>
class DepthwiseConv : public KernelLite<TARGET(kLoongArch), Ptype> {
 public:
  DepthwiseConv() = default;
  ~DepthwiseConv() {}
  void PrepareForRun() override;
  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDepthwise"};
#define PROFILE_INFO(dtype1, dtype2)                                        \
  template <>                                                               \
  void DepthwiseConv<PRECISION(dtype1), PRECISION(dtype2)>::                \
      SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;                               \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;

#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

 private:
  using param_t = operators::ConvParam;
  Tensor input_pack_;
  Tensor input_padding_;
  Tensor filter_pack_;
  Tensor output_pack_;
  bool flag_trans_bias_{true};
  std::vector<float> w_scale_;
  Tensor bias_;
};

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
