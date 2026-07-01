#pragma once
#include <cstddef>
#include <cstdint>

namespace llm_engine {
    float dot_product_q8_0(const int8_t* a, const float* b, size_t n);
    void dequantize_block_q2_k(const uint8_t* src, float* dst, size_t dst_size);
}