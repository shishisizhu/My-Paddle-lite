/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include "lite/backends/loongarch/math/instruction_utils.h"
#include <lsxintrin.h>
#include <lasxintrin.h>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

#define loadu_ps(a) (__m256)__lasx_xvld(a, 0)
#define fmadd_ps(a, b, c) __lasx_xvfmadd_s(a, b, c)
#define storeu_ps(a, b) __lasx_xvst(b, a, 0)
#define setzero_ps() (__m256)__lasx_xvreplgr2vr_w(0)
#define max_ps(a, b) __lasx_xvfmax_s(a, b)
#define min_ps(a, b) __lasx_xvfmin_s(a, b)
#define set1_ps(a) (__m256)__lasx_xvreplgr2vr_w(*reinterpret_cast<const int*>(&a))
#define mul_ps(a, b) __lasx_xvfmul_s(a, b)
#define cmp_ps(a, b, c) lasx_m256_cmp_ps(a, b, c)
#define blendv_ps(a, b, c) (__m256)lasx_m256i_blendv_ps(a, b, c)
#define add_ps(a, b) __lasx_xvfadd_s(a, b)
#define block_channel 8
#define Type __m256


}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
