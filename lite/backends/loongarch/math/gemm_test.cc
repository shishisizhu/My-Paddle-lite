#include "lite/backends/loongarch/math/gemm_s8u8_pack.h"
#include <lasxintrin.h>
#include <lsxintrin.h>
#include <iostream>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
  void test_gemm() {
    int8_t array[1000];
    for(int i=0; i<1000; i++)
            array[i]= i;
    int8_t out[2000];
    gemm_s8u8s8_prepackA(5, 5, array, out, false);
    for(int i=0; i<2000; i++)
        std::cout<< (int)out[i]<<" ";
    std::cout<<std::endl;
  }

}
}
}
}
int main () {
    paddle::lite::loongarch::math::test_gemm();
}
