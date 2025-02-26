#pragma once

#include <stdint.h>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

#define TRANS_INT8_UINT8_OFFT (128)

// PackA 's K dim need 4-aligned,
// so it needs M * K_4aligned Bytes.
void gemm_s8u8s8_prepackA(
    int M, int K, const int8_t* A, int8_t* pack_A, bool is_trans);

void gemm_s8u8s8_runpackB(
    int N, int K, int stride, const int8_t* B, uint8_t* pack_B, bool is_trans);

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
