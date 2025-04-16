#pragma once
#include <lsxintrin.h>
#include <lasxintrin.h>
#include <iostream>
#include <cstring>

//lasx_m256i_set_epi32

#define CONVERT_IMM8(x) ((x) ^ (2U | (2U << 4)))

#ifdef Q_CC_CLANG
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
// Convert __m128i to __m256i
static inline __m256i lasx_cvtm128i(__m128i in)
{
    __m256i out = __lasx_xvldi(0);
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX"\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "+f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert two __m128i to __m256i
static inline __m256i lasx_set_q(__m128i inhi, __m128i inlo)
{
    __m256i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".ifnc %[out], %[hi]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
        "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out), [hi] "+f" (inhi)
        : [lo] "f" (inlo)
    );
    return out;
}
// Convert __m256i low part to __m128i
static inline __m128i lasx_extracti128_lo(__m256i in)
{
    __m128i out;
    __asm__ volatile (
        ".ifnc %[out], %[in]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    vori.b $vr\\i, $vr\\j, 0        \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert __m256i high part to __m128i
static inline __m128i lasx_extracti128_hi(__m256i in)
{
    __m128i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}

inline void printvector(__m128 mask) {
    std::cout<< "------printstart-------" << std::endl;
    float ptr1[4];
    __lsx_vst(mask, ptr1, 0);
    uint32_t ptr[4] = {0};
    __lsx_vst(mask, ptr, 0);

    std::cout<<"float: "<<std::endl;
    for(int i=0; i<4; i++) 
        std::cout<< ptr1[i]<<" ";
    std::cout<< std::endl;

    std::cout<<"hex: "<<std::endl;
    for(int i=0; i<4; i++) 
        std::cout<< std::hex<<ptr[i]<<" ";
std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline void printvector(__m128i mask) {
    std::cout<< "------printstart-------" << std::endl;
    float ptr1[4];
    __lsx_vst(mask, ptr1, 0);
    uint32_t ptr[4] = {0};
    __lsx_vst(mask, ptr, 0);

    std::cout<<"float: "<<std::endl;
    for(int i=0; i<4; i++) 
        std::cout<< ptr1[i]<<" ";
    std::cout<< std::endl;

    std::cout<<"hex: "<<std::endl;
    for(int i=0; i<4; i++) 
        std::cout<<ptr[i]<<" ";
std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline void printvector_8i(__m128i mask) {
    std::cout<< "------printstart-------" << std::endl;
    int8_t ptr1[16];
    __lsx_vst(mask, ptr1, 0);
    
    for(int i=0; i<16; i++) 
        std::cout<<(int)ptr1[i]<<" ";
    std::cout<< std::endl;

std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline void printvector_8i(__m256i mask) {
    std::cout<< "------printstart-------" << std::endl;
    int8_t ptr1[32];
    __lasx_xvst(mask, ptr1, 0);
    
    for(int i=0; i<32; i++) 
        std::cout<< std::hex<< (int16_t)ptr1[i]<<" ";
    std::cout<< std::endl;

std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline void printvector(__m256 mask) {
    std::cout<< "------printstart-------" << std::endl;
    float ptr1[8];
    __lasx_xvst(mask, ptr1, 0);
    uint32_t ptr[8] = {0};
    __lasx_xvst(mask, ptr, 0);

    std::cout<<"float: "<<std::endl;
    for(int i=0; i<8; i++) 
        std::cout<< ptr1[i]<<" ";
    std::cout<< std::endl;

    std::cout<<"hex: "<<std::endl;
    for(int i=0; i<8; i++) 
        std::cout<< std::hex<<ptr[i]<<" ";
std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline void printvector(__m256i mask) {
    std::cout<< "------printstart-------" << std::endl;
    float ptr1[8];
    __lasx_xvst(mask, ptr1, 0);
    uint32_t ptr[8] = {0};
    __lasx_xvst(mask, ptr, 0);

    std::cout<<"float: "<<std::endl;
    for(int i=0; i<8; i++) 
        std::cout<< ptr1[i]<<" ";
    std::cout<< std::endl;

    std::cout<<"hex: "<<std::endl;
    for(int i=0; i<8; i++) 
        std::cout<< std::hex<<ptr[i]<<" ";
    std::cout<<std::endl;
    std::cout<< "------printend-------" << std::endl<<std::endl;
}

inline __m256i lasx_m256i_set_epi32(int a7, int a6, int a5, int a4, int a3, int a2, int a1, int a0) {
    __m256i vec = __lasx_xvreplgr2vr_w(a0); // create 
    vec = __lasx_xvinsgr2vr_w(vec, a1, 1);  
    vec = __lasx_xvinsgr2vr_w(vec, a2, 2); 
    vec = __lasx_xvinsgr2vr_w(vec, a3, 3); 
    vec = __lasx_xvinsgr2vr_w(vec, a4, 4);  
    vec = __lasx_xvinsgr2vr_w(vec, a5, 5);  
    vec = __lasx_xvinsgr2vr_w(vec, a6, 6);  
    vec = __lasx_xvinsgr2vr_w(vec, a7, 7);  
    return vec;
}

inline __m256 lasx_m256_set_ps(float a7, float a6, float a5, float a4, float a3, float a2, float a1, float a0) {
    __m256i vec = __lasx_xvreplgr2vr_w(a0); 
    vec = __lasx_xvinsgr2vr_w(vec, a1, 1);  
    vec = __lasx_xvinsgr2vr_w(vec, a2, 2); 
    vec = __lasx_xvinsgr2vr_w(vec, a3, 3); 
    vec = __lasx_xvinsgr2vr_w(vec, a4, 4);  
    vec = __lasx_xvinsgr2vr_w(vec, a5, 5);  
    vec = __lasx_xvinsgr2vr_w(vec, a6, 6);  
    vec = __lasx_xvinsgr2vr_w(vec, a7, 7);  
    return (__m256)vec;
}
inline __m128i lsx_m128i_setr_epi32(int a0, int a1, int a2, int a3) {
    __m128i vec = __lsx_vreplgr2vr_w(a0); 
    vec = __lsx_vinsgr2vr_w(vec, a1, 1);  
    vec = __lsx_vinsgr2vr_w(vec, a2, 2); 
    vec = __lsx_vinsgr2vr_w(vec, a3, 3); 
    return vec;  
}

inline __m128i lsx_128im_set_epi32(int a3, int a2, int a1, int a0) {
    __m128i vec = __lsx_vreplgr2vr_w(a0); 
    vec = __lsx_vinsgr2vr_w(vec, a1, 1);  
    vec = __lsx_vinsgr2vr_w(vec, a2, 2); 
    vec = __lsx_vinsgr2vr_w(vec, a3, 3); 
    return vec;  
}


inline __m256i lasx_m256i_blendv_ps(__m256i a, __m256i b, __m256i mask) {

    __m256i mask_bits = __lasx_xvsrai_w(mask, 31);

    __m256i result_i = __lasx_xvbitsel_v(a, b, mask_bits);

    return result_i;
}
inline __m256i lasx_m256i_blendv_ps(__m256 a, __m256i b, __m256i mask) {

    __m256i mask_bits = __lasx_xvsrai_w(mask, 31);

    __m256i result_i = __lasx_xvbitsel_v((__m256i)a, b, mask_bits);

    return result_i;
}

inline __m256i lasx_m256i_blendv_ps(__m256 a, __m256 b, __m256 mask) {

    __m256i mask_bits = __lasx_xvsrai_w((__m256i)mask, 31);

    __m256i result_i = __lasx_xvbitsel_v((__m256i)a, (__m256i)b, mask_bits);

    return result_i;
}

inline __m256i lasx_m256i_blendv_ps(__m256 a, __m256 b, __m256i mask) {

    __m256i mask_bits = __lasx_xvsrai_w(mask, 31);

    __m256i result_i = __lasx_xvbitsel_v((__m256i)a, (__m256i)b, mask_bits);

    return result_i;
}

inline __m256i lasx_m256i_blendv_ps(__m256i a, __m256 b, __m256i mask) {

    __m256i mask_bits = __lasx_xvsrai_w(mask, 31);

    __m256i result_i = __lasx_xvbitsel_v(a, (__m256i)b, mask_bits);

    return result_i;
}


inline __m256 lasx_m256_blendv_ps(__m256 a, __m256 b, __m256 mask) {

    __m256i a_i = (__m256i)a;
    __m256i b_i = (__m256i)b;
    __m256i mask_i = (__m256i)mask;

    __m256i mask_bits = __lasx_xvsrai_w(mask_i, 31);
    __m256i result_i = __lasx_xvbitsel_v(a_i, b_i, mask_bits);

    return (__m256)result_i;
}

inline void lasx_void_maskstore_ps(void* ptr, __m256i mask, __m256 data) {

    __m256i full_mask = __lasx_xvsrai_w(mask, 31);
    __m256i old_data = __lasx_xvld(ptr, 0);
    __m256i blended = __lasx_xvbitsel_v(old_data, (__m256i)data, full_mask);
    __lasx_xvst(blended, ptr, 0);
}

inline void lasx_maskstore_epi32(void * ptr, __m256i mask, __m256i mask_bits) {
     __m256i full_mask = __lasx_xvsrai_w(mask, 31);
     __m256i old_data = __lasx_xvld(ptr, 0);
     __m256i blended = __lasx_xvbitsel_v(old_data, data, full_mask);
    __lasx_xvst(blended, ptr, 0);
}

inline void lsx_void_maskstore_ps(void* ptr, __m128i mask, __m128 data) {
    __m128i full_mask = __lsx_vsrai_w(mask, 31);
    __m128i old_data = __lsx_vld(ptr, 0);
    __m128i blended = __lsx_vbitsel_v(old_data, (__m128i)data, full_mask);
    __lsx_vst(blended, ptr, 0);
}


inline __m128 lsx_m128_blendv_ps(__m128 a, __m128 b, __m128 mask) {

    __m128i mask_bits = __lsx_vsrai_w((__m128i)mask, 31);
    __m128i result_i = __lsx_vbitsel_v((__m128i)a, (__m128i)b, mask_bits);

    return (__m128)result_i;
}

inline __m128 lsx_m128_blendv_ps(__m128 a, __m128 b, __m128i mask) {

    __m128i mask_bits = __lsx_vsrai_w(mask, 31);
    __m128i result_i = __lsx_vbitsel_v((__m128i)a, (__m128i)b, mask_bits);

    return (__m128)result_i;
}

inline __m256i lasx_mm_shuffle(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    
    uint32_t nums[8] = {a0+4, a1+4, a2, a3, a0+4, a1+4, a2, a3};
    return __lasx_xvld(nums, 0);
}
inline __m256i lasx_mm_shuffle(uint32_t a) {
    uint32_t a0 = a&0x3, a1 = (a>>2)&0x3, a2 = (a>>4)&0x3, a3 = (a>>6)&0x3;
    uint32_t nums[8] = {a0+4, a1+4, a2, a3, a0+4, a1+4, a2, a3};
    return __lasx_xvld(nums, 0);
}
inline __m128i lsx_mm_shuffle(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) {
    
    uint32_t nums[4] = {a0+4, a1+4, a2, a3};
    return __lsx_vld(nums, 0);
}

inline __m128i lsx_mm_shuffle(uint32_t a) {
    uint32_t a0 = a&0x3, a1 = (a>>2)&0x3, a2 = (a>>4)&0x3, a3 = (a>>6)&0x3;
    uint32_t nums[4] = {a0+4, a1+4, a2, a3};
    return __lsx_vld(nums, 0);
}
inline __m256i lasx_m256i_shuffle_ps(__m256i b, __m256i c, __m256i a) {
    return __lasx_xvshuf_w(a, b, c);
}
// function: input-4x8, output-8x4
inline __m128i lsx_m128i_shuffle_ps(__m128i b, __m128i c, __m128i a) {
    return __lsx_vshuf_w(a, b, c);
}

inline __m256 lasx_m256_cmp_ps(__m256 a, __m256 b, int c) {
    switch (c) {
        case 2:
            return (__m256)__lasx_xvfcmp_sle_s(a, b);
        default:
            std::cout<< "In instruction_utils.cc Error!" << std::endl;
    }
    std::cout<< "In instruction_utils.cc Error!" << std::endl;
    return a;
}

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) { \
    __m128i tmp3, tmp2, tmp1, tmp0; \
    tmp0 = lsx_m128i_shuffle_ps((__m128i)(row0), (__m128i)(row1), lsx_mm_shuffle(0x44)); \
    tmp2 = lsx_m128i_shuffle_ps((__m128i)(row0), (__m128i)(row1), lsx_mm_shuffle(0xEE)); \
    tmp1 = lsx_m128i_shuffle_ps((__m128i)(row2), (__m128i)(row3), lsx_mm_shuffle(0x44)); \
    tmp3 = lsx_m128i_shuffle_ps((__m128i)(row2), (__m128i)(row3), lsx_mm_shuffle(0xEE)); \
    (row0) = lsx_m128i_shuffle_ps(tmp0, tmp1, lsx_mm_shuffle(0x88)); \
    (row1) = lsx_m128i_shuffle_ps(tmp0, tmp1, lsx_mm_shuffle(0xDD)); \
    (row2) = lsx_m128i_shuffle_ps(tmp2, tmp3, lsx_mm_shuffle(0x88)); \
    (row3) = lsx_m128i_shuffle_ps(tmp2, tmp3, lsx_mm_shuffle(0xDD)); \
}

inline __m128i lsx_shuffle_epi8(__m128i a, __m128i mask) {
    // 提取控制字节的低四位（索引部分）
    __m128i mask_low4 = __lsx_vand_v(mask, __lsx_vreplgr2vr_b(0x0F));
    
    // 执行字节重排，a作为两个输入源以覆盖0-15索引
    __m128i shuffled = __lsx_vshuf_b(a, a, mask_low4);
    
    // 检测mask中的最高位，生成条件掩码
    __m128i high_bit = __lsx_vand_v(mask, __lsx_vreplgr2vr_b(0x80));
    __m128i cmp = __lsx_vseq_b(high_bit, __lsx_vreplgr2vr_b(0x80));
    
    // 取反条件掩码以准备清零操作
    __m128i inv_mask = __lsx_vxor_v(cmp, __lsx_vreplgr2vr_b(0xFF));
    
    // 应用掩码，将高位对应的位置清零
    __m128i result = __lsx_vand_v(shuffled, inv_mask);
    
    return result;
}
//按参数给出vec
inline __m128i lsx_set_epi8(int8_t a15, int8_t a14 ,int8_t a13, int8_t a12, int8_t a11, int8_t a10,int8_t a9,int8_t a8,int8_t a7, int8_t a6, int8_t a5, int8_t a4, int8_t a3, int8_t a2, int8_t a1, int8_t a0) {
    int8_t data[16] {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    return __lsx_vld(data, 0);
}


inline __m128i lsx_packs_epi32(__m128i a, __m128i b) {
    // 1. 对每个元素进行饱和操作，限制为 int16 范围
    __m128i a_sat = __lsx_vsat_w(a, 15);  // 15 位符号位（16 位有符号）
    __m128i b_sat = __lsx_vsat_w(b, 15);
    //printvector(a_sat);printvector(b_sat);
    // 3. 打包为 8 个 16 位元素（交错低半部和高半部）
    __m128i res = __lsx_vpickev_h(b_sat, a_sat);
    return res;
}
inline __m256i lasx_m256i_packs_epi32(__m256i a, __m256i b) {
   __m256i a_sat = __lasx_xvsat_w(a, 15);
   __m256i b_sat = __lasx_xvsat_w(b, 15);
   __m256i res = __lasx_xvpickev_h(b_sat, a_sat);
   return res;
}

inline __m128i lsx_packs_epi16(__m128i a, __m128i b) {
    // 1. 对每个元素进行饱和操作，限制为 int16 范围
    __m128i a_sat = __lsx_vsat_h(a, 7);  // 15 位符号位（16 位有符号）
    __m128i b_sat = __lsx_vsat_h(b, 7);
   // printvector(a_sat);printvector(b_sat);
    // 3. 打包为 8 个 16 位元素（交错低半部和高半部）
    __m128i res = __lsx_vpickev_b(b_sat, a_sat);
    return res;
}

inline __m256i lasx_packs_epi16(__m256i a, __m256i b) {
    __m256i a_sat = __lasx_xvsat_h(a, 7);
    __m256i b_sat = __lasx_xvsat_h(b, 7);
    __m256i res = __lasx_xvpickev_b(b_sat, a_sat);
    return res;
}
inline __m256i lasx_maddubs_epi16(__m256i a, __m256i b) {
    return __lasx_xvsat_h(__lasx_xvmaddwev_h_bu_b(__lasx_xvmulwod_h_bu_b(a, b), a, b), 15);
}

inline __m256i lasx_madd_epi16(__m256i a, __m256i b) {
    return __lasx_xvsat_w(__lasx_xvmaddwev_w_h(__lasx_xvmulwod_w_h(a, b), a, b), 31);
}

inline __m128i lsx_maddubs_epi16(__m128i a, __m128i b) {
    return __lsx_vsat_h(__lsx_vmaddwev_h_bu_b(__lsx_vmulwod_h_bu_b(a, b), a, b), 15);
}

inline __m128i lsx_madd_epi16(__m128i a, __m128i b) {
    return __lsx_vsat_w(__lsx_vmaddwev_w_h(__lsx_vmulwod_w_h(a, b), a, b), 31);
}

inline __m256i lasx_hadd_epi32(__m256i a, __m256i b) {
    __m256i enres = __lasx_xvpickev_w(b, a);
    __m256i odres = __lasx_xvpickod_w(b, a);
    return __lasx_xvadd_w(enres, odres);
}

inline __m128i lsx_hadd_h(__m128i a, __m128i b)
{
    __m128i tmp1 = __lsx_vpickev_h(b, a);
    __m128i tmp2 = __lsx_vpickod_h(b, a);
    return __lsx_vadd_h(tmp1, tmp2);
}

inline __m128i lsx_hadd_w(__m128i a, __m128i b)
{
    __m128i tmp1 = __lsx_vpickev_w(b, a);
    __m128i tmp2 = __lsx_vpickod_w(b, a);
    return __lsx_vadd_w(tmp1, tmp2);
}

inline __m128 lsx_hadd_s(__m128 a, __m128 b)
{
    __m128 tmp1 = (__m128)__lsx_vpickev_w((__m128i)b, (__m128i)a);
    __m128 tmp2 = (__m128)__lsx_vpickod_w((__m128i)b, (__m128i)a);

    return __lsx_vfadd_s(tmp1, tmp2);
}

inline __m128i lsx_srli_si128(__m128i a, int imm) {

    // 生成控制掩码
    uint8_t mask[16];
    for (int i = 0; i < 16; i++) {
        mask[i] = (i < 16-imm) ? imm+i : 64; // 0x80 表示填充 0
    }

    // 加载掩码到向量寄存器
    __m128i vmask = __lsx_vld(mask, 0);

    // 使用字节重排指令实现右移
    return __lsx_vshuf_b(a, a, vmask);
}

__m128i lsx_loadl_pi(__m128i a, void *mem_addr) {
    // 将__m64指针转换为long long类型读取
    long long tmp = *(const long long *)mem_addr;
    // 将tmp插入到向量a的低64位，高位保持不变
    return __lsx_vinsgr2vr_d(a, tmp, 0);
}

inline __m128i lsx_loadl_epi64(const void* memdata) {
    long int data;
    memcpy(&data, memdata, sizeof(data));
    __m128i zero = __lsx_vldi(0);
    return __lsx_vinsgr2vr_d(zero, data, 0);
}

inline __m256i lasx_maskload_epi32(int const* mem_addr, __m256i mask) {
    // 将掩码向量的符号位扩展为全0或全1
    __m256i load_mask = __lasx_xvsrai_w(mask, 31);
    // 加载内存中的数据（8个int32）
    __m256i data = __lasx_xvld(mem_addr, 0);
    // 应用掩码，保留需要的数据，其余置0
    __m256i result = __lasx_xvand_v(data, load_mask);
    return result;
}

inline __m256i lasx_maskload_epi64(void * mem_addr, __m256i mask) {
    // 将掩码向量的符号位扩展为全0或全1
    __m256i load_mask = __lasx_xvsrai_d(mask, 63);

    __m256i data = __lasx_xvld(mem_addr, 0);
    // 应用掩码，保留需要的数据，其余置0
    __m256i result = __lasx_xvand_v(data, load_mask);
    return result;
}

__m256i lasx_inserti128_si256(__m256i a, __m128i b, const int imm8) {
   return imm8==0 ? lasx_set_q(lasx_extracti128_hi(a), b) : lasx_set_q(b, lasx_extracti128_lo(a));
}



