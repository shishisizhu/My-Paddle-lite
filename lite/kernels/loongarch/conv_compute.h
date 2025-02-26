#pragma once
//TODO
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"
//TODO end

#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch{

template <PrecisionType Ptype, PrecisionType OutType>
class ConvCompute : public KernelLite<TARGET(kLoongArch), Ptype> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    CHECK(impl_);
    impl_->ReInitWhenNeeded();
  }

  virtual void Run() {
    CHECK(impl_);
    impl_->Run();
  }

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    impl_->SetProfileRuntimeKernelInfo(ch);
  }
#endif

  ~ConvCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  using param_t = operators::ConvParam;
  KernelLite<TARGET(kLoongArch), Ptype>* impl_{nullptr};
};

    
}//namespace loongarch
}//namespace kernels
}//namespace lite
}//namespace paddle
