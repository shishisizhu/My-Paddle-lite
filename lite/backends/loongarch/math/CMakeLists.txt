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
)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}) 
add_executable(
    test_gemm 
    gemm_s8u8_pack.cc 
    gemm_test.cc 
)
 # 链接库 
target_link_libraries(test_gemm 
    ${PADDLE_LITE_ROOT}/build.lite.linux.loongarch.gcc/inference_lite_lib/cxx/lib/libpaddle_api_light_bundled.a 
    ${GTEST_LIBRARIES} pthread ) 
#enable_testing() #add_test(NAME test_depthwise COMMAND test_depthwise_conv)
