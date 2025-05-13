if(LITE_WITH_LOONGARCH AND NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mlasx")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mlasx")
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  set(lite_kernel_deps ${lite_kernel_deps} loongarch_math CACHE INTERNAL "")
  set_source_files_properties (${lite_kernel_deps} PROPERTIES COMPILE_FLAGS "-mlasx")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../backends/loongarch/math)
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()
message(STATUS "compile with lite LoongArch kernels")
#for conv op
  add_kernel(conv_depthwise_loongarch LoongArch basic SRCS conv_depthwise.cc)
  add_kernel(conv_compute_loongarch LoongArch basic SRCS conv_compute.cc)

  
