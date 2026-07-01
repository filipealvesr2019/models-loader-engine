#pragma once
#include <cstddef>
#include <cstdint>

namespace llm_engine {
    float dot_product_q8_0(const int8_t* a, const float* b, size_t n);
    void dequantize_block_q2_k_scalar(const uint8_t* src, float* dst, size_t dst_size);
    void dequantize_block_q2_k(const uint8_t* src, float* dst, size_t dst_size);
    void rms_norm(const float* x, float* y, size_t n, float eps = 1e-5f);
    void add_vectors(const float* a, const float* b, float* out, size_t n);
}