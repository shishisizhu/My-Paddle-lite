cmake_minimum_required(VERSION 3.10)
project(test_conv_utils)
add_compile_options(-mlasx) 
add_definitions(-DLITE_WITH_LOONGARCH)
set(PADDLE_LITE_ROOT "/home/loongson/Paddle-Lite")
 # 链接目录 
link_directories( 
    ${PADDLE_LITE_ROOT}/build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/lib 
)
 # 包含目录配置 
include_directories( 
        ${PADDLE_LITE_ROOT} 
        ${PADDLE_LITE_ROOT}/lite
        ${PADDLE_LITE_ROOT}/lite/backends 
        ${PADDLE_LITE_ROOT}/lite/backends/loongarch 
        ${PADDLE_LITE_ROOT}/lite/backends/loongarch/math
        ${PADDLE_LITE_ROOT}/lite/backends/loongarch/fluid 
)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}) 
add_executable(test_conv5x5
    vol2col.cc
    instance_norm.cc
    im2col.cc
    gemm_s8u8_compute.cc
    gemm_s8u8_kernel.cc
    gemm_s8u8_pack.cc
    blas.cc
    blas.h
    conv_bias.cc
    conv_utils.cc
    fill_bias_activate.cc
    conv3x3s1_depthwise_int8.cc
    conv3x3s2_depthwise_int8.cc
    conv_depthwise_pack8.cc
    conv_depthwise_pack4.cc
    conv_depthwise_3x3.cc
    conv_depthwise_5x5.cc
    con_depthwise_test.cc 
)
 # 链接库 
target_link_libraries(test_conv5x5
    ${PADDLE_LITE_ROOT}/build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/lib/libpaddle_api_light_bundled.a 
    ${GTEST_LIBRARIES} pthread ) 
#enable_testing() #add_test(NAME test_depthwise COMMAND test_depthwise_conv)
