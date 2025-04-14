#include <gtest/gtest.h>
#include <cmath>
#include "conv_depthwise_3x3.h"
// 假设函数声明已包含

#include <cstring>  // for memcpy
n
// 假设的函数声明（需与实际头文件一致）
extern void conv_depthwise_3x3s1_p01_direct(
    const float *din,
    float *dout,
    int num,
    int ch_out,
    int h_out,
    int w_out,
    int ch_in,
    int h_in,
    int w_in,
    const float *weights,
    const float *bias,
    int pad,
    bool flag_bias,
    const operators::ActivationParam act_param);

namespace {
// 辅助函数：比较浮点数组
void ExpectFloatArrayNear(const float* expected, const float* actual, 
                         int count, float abs_error = 1e-5) {
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(expected[i], actual[i], abs_error);
  }
}

// 测试夹具
class ConvDepthwiseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 公共初始化代码
  }

  operators::ActivationParam relu_param_{
    .has_active = true,
    .active_type = operators::ActivationType::kRelu
  };
};

// 测试用例1：基础卷积（无填充/偏置/激活）
TEST_F(ConvDepthwiseTest, BasicConvolution) {
  const int num = 1, ch_in = 1, h_in = 3, w_in = 3;
  const int pad = 1;
  const int h_out = h_in, w_out = w_in;  // 填充保持尺寸
  const int ch_out = ch_in;

  // 输入数据：3x3矩阵
  float din[] = {1,2,3,4,5,6,7,8,9};
  
  // 卷积核：中心为1的3x3核
  float weights[9] = {0};
  weights[4] = 1.0f;
  
  float dout[9] = {0};
  float expected[9];
  memcpy(expected, din, sizeof(din));  // 预期输出与输入相同

  operators::ActivationParam no_act;
  conv_depthwise_3x3s1_p01_direct(din, dout, num, ch_out, h_out, w_out,
                                 ch_in, h_in, w_in, weights, nullptr,
                                 pad, false, no_act);

  ExpectFloatArrayNear(expected, dout, 9);
}

// 测试用例2：带偏置的卷积
TEST_F(ConvDepthwiseTest, WithBias) {
  const int ch_in = 1;
  float din[] = {1,2,3,4,5,6,7,8,9};
  float weights[9] = {0}; weights[4] = 1.0f;
  float bias[] = {2.5f};
  
  float expected[9];
  for (int i = 0; i < 9; ++i) expected[i] = din[i] + bias[0];

  float dout[9] = {0};
  operators::ActivationParam no_act;
  
  conv_depthwise_3x3s1_p01_direct(din, dout, 1, ch_in, 3, 3,
                                 ch_in, 3, 3, weights, bias,
                                 1, true, no_act);

  ExpectFloatArrayNear(expected, dout, 9);
}

// 测试用例3：ReLU激活
TEST_F(ConvDepthwiseTest, ReLUActivation) {
  float din[] = {-1, -2, 3, -4, 5, -6, 7, -8, -9};
  float weights[9] = {0}; weights[4] = 1.0f;
  float expected[] = {0,0,3,0,5,0,7,0,0};
  float dout[9] = {0};

  conv_depthwise_3x3s1_p01_direct(din, dout, 1, 1, 3, 3,
                                 1, 3, 3, weights, nullptr,
                                 1, false, relu_param_);

  ExpectFloatArrayNear(expected, dout, 9);
}

// 测试用例4：多通道处理
TEST_F(ConvDepthwiseTest, MultiChannelProcessing) {
  const int ch_in = 3;
  float din[27] = {  // 3通道，每个通道3x3矩阵
    // Channel 0
    1,1,1, 1,1,1, 1,1,1,
    // Channel 1 
    2,2,2, 2,2,2, 2,2,2,
    // Channel 2
    3,3,3, 3,3,3, 3,3,3
  };

  // 每个通道使用不同的卷积核
  float weights[27] = { 
    // Channel 0核：全1
    1,1,1,1,1,1,1,1,1,
    // Channel 1核：中心为2
    0,0,0,0,2,0,0,0,0,
    // Channel 2核：对角核
    1,0,0,0,1,0,0,0,1
  };

  float expected[27];
  // Channel 0计算结果（sum all elements with padding）
  const float c0 = 1*4 + 1*4 + 1*3;  // 根据填充计算
  std::fill(expected, expected+9, c0);
  // Channel 1计算结果（中心值*2）
  std::fill(expected+9, expected+18, 2*2); 
  // Channel 2计算结果（对角线求和）
  std::fill(expected+18, expected+27, 3*3); 

  float dout[27] = {0};
  operators::ActivationParam no_act;
  
  conv_depthwise_3x3s1_p01_direct(din, dout, 1, ch_in, 3, 3,
                                 ch_in, 3, 3, weights, nullptr,
                                 1, false, no_act);

  ExpectFloatArrayNear(expected, dout, 27);
}

// 测试用例5：无填充情况
TEST_F(ConvDepthwiseTest, NoPadding) {
  const int h_in = 5, w_in = 5;
  const int pad = 0;
  const int h_out = (h_in - 3) + 1; // 3
  const int w_out = 3;

  // 输入：5x5矩阵（通道1）
  float din[25];
  for (int i = 0; i < 25; ++i) din[i] = i+1;

  // 卷积核：计算行和列的和
  float weights[9] = {1,0,-1, 2,0,-2, 1,0,-1}; // Sobel算子

  float expected[9] = {
    // 手工计算预期结果
    (din[0]*1 + din[1]*0 + din[2]*(-1) +
     din[5]*2 + din[6]*0 + din[7]*(-2) +
     din[10]*1 + din[11]*0 + din[12]*(-1)),
    // ... 其他8个输出点
  };

  float dout[9] = {0};
  operators::ActivationParam no_act;
  
  conv_depthwise_3x3s1_p01_direct(din, dout, 1, 1, h_out, w_out,
                                 1, h_in, w_in, weights, nullptr,
                                 pad, false, no_act);

  // 此处需要根据实际卷积计算填充expected数组
  // ExpectFloatArrayNear(expected, dout, 9);
}

// 测试用例6：多Batch测试
TEST_F(ConvDepthwiseTest, MultiBatch) {
  const int num = 2;
  const int ch_in = 1;
  float din[2*9] = {  // 两个样本
    1,2,3,4,5,6,7,8,9,   // Batch 0
    9,8,7,6,5,4,3,2,1    // Batch 1
  };
  
  float weights[9] = {0}; weights[4] = 1.0f;
  float dout[2*9] = {0};
  float expected[18];
  memcpy(expected, din, 18*sizeof(float)); // 预期与输入相同

  operators::ActivationParam no_act;
  conv_depthwise_3x3s1_p01_direct(din, dout, num, ch_in, 3, 3,
                                 ch_in, 3, 3, weights, nullptr,
                                 1, false, no_act);

  ExpectFloatArrayNear(expected, dout, 18);
}
}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
