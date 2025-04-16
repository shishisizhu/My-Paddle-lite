
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

void fill_bias_act(float* tensor,
                   const float* bias,
                   int channel,
                   int channel_size,
                   bool flag_bias,
                   const operators::ActivationParam* act_param);

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
