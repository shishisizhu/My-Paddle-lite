#include "lite/backends/loongarch/math/conv_depthwise_impl.h"
#include "lite/backends/loongarch/math/conv_depthwise_5x5.h"
#include "lite/backends/loongarch/math/conv_depthwise_int8.h"
#include <lasxintrin.h>
#include <lsxintrin.h>
#include <iostream>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
  void test_conv_dp5x5s1() {
    float din[25];
    int count = 0.5;
    for(int i=0; i<25; i++)
        din[i] = count++;
    float weights[25];
    for(int i=0; i<25; i++)
        weights[i] = 0;
    weights[12] = 1;
    operators::ActivationParam act_param;
    float out[25];
    conv_depthwise_5x5s1(din, out, 1, 1, 3, 3, 1, 5, 5, weights, NULL, 1, false, act_param);
    for(int i=0; i<9; i++)
        std::cout<< out[i] << " ";
    std::cout<<std::endl;
    
  }
    
  void test_conv_dp3x3s1() {
    float din[25];
    float count = 0;
    for(int i=0; i<25; i++)
        din[i] = count++;
    float weights[9];
    for(int i=0; i<9; i++)
        weights[i] = 0;
    weights[4] = 1;
    operators::ActivationParam act_param;
    float out[25];
    conv_depthwise_3x3s1_p01_direct(din, out, 1, 1, 3, 3, 1, 5, 5, weights, NULL, 0, false, act_param);
    for(int i=0; i<25; i++)
        std::cout<< out[i] << " ";
    std::cout<<std::endl;
    
  }

  void test_conv_dp3x3s1_int8() {
    int8_t din[25];
    float bias[100];
    for(int i=0; i<100; i++)
        bias[i] = 1.0;
    float scale[100];
    for(int i=0; i<100; i++)
        scale[i] = 1.0;
    int count = 0;
    for(int i=0; i<25; i++)
        din[i] = count++;
    int8_t weights[9];
    for(int i=0; i<9; i++)
        weights[i] = 0;
    weights[4] = 1;
    operators::ActivationParam act_param;
    float out[100];
    conv_3x3s1_dw_int8(out, din, weights, bias, 1, 1, 5, 5, 3, 3, 0, 0, false, 0.1, scale, new LoongArchContext());
    for(int i=0; i<9; i++)
        std::cout<< out[i] << " ";
    std::cout<<std::endl;
    
  }

}
}
}
}
int main () {
    paddle::lite::loongarch::math::test_conv_dp3x3s1_int8();
}
