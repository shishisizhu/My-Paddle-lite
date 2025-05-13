#include <lasxintrin.h>
#include <lsxintrin.h>
#include "lite/backends/loongarch/math/conv_utils.h"
#include <algorithm>
namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
  void test_transpose8() {
      float count = 0;
      float array[8][8];
      for(int i=0; i<8; i++) {
        for(int j=0; j<8; j++) {
            array[i][j] = count++;
        }
      }
      __m256 vecs[8];
      for(int i=0; i<8; i++)
        vecs[i] = (__m256)__lasx_xvld(array[i], 0);
      transpose8_ps(vecs[0], vecs[1], vecs[2], vecs[3],vecs[4],vecs[5],vecs[6],vecs[7]);
      for(int i=0; i<8; i++)
        printvector(vecs[i]);
  }

      void test_transpose4x8() {
      float count = 0;
      float array[4][8];
      for(int i=0; i<4; i++) {
        for(int j=0; j<8; j++) {
            array[i][j] = count++;
        }
      }
      __m256 vecs[4];
      for(int i=0; i<4; i++)
        vecs[i] = (__m256)__lasx_xvld(array[i], 0);
      //transpose4x8_ps(vecs[0], vecs[1], vecs[2], vecs[3]);
      for(int i=0; i<4; i++)
        printvector(vecs[i]);
  }

      void test_transpose4() {
      float count = 0;
      float array[4][4];
      for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            array[i][j] = count++;
        }
      }
      __m128 vecs[4];
      for(int i=0; i<4; i++)
        vecs[i] = (__m128)__lsx_vld(array[i], 0);
      _MM_TRANSPOSE4_PS(vecs[0], vecs[1], vecs[2], vecs[3]);
      for(int i=0; i<4; i++)
        printvector(vecs[i]);
  }
    
    void test_activation() {
        float data[8];
        for(int i=0; i<8; i++)
            data[i] = i * (1 - (i%2?0 : 2));
        __m256 vec = (__m256)__lasx_xvld(data, 0);
        operators::ActivationParam act_param;
        lite_api::ActivationType act_type = lite_api::ActivationType::kLeakyRelu;
       act_param.Leaky_relu_alpha = 0.01;
        act_param.has_active = true;
        vec = activation8_m256(vec, act_type, act_param);
        float res[8];
        __lasx_xvst(vec, res, 0);
        for(int i=0; i<8; i++) 
            std::cout<<res[i]<<" ";
        std::cout<<std::endl;
    }
}
}
}
}
int main() {
    paddle::lite::loongarch::math::test_activation();
}
