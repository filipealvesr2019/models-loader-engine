#pragma once
#include <cstddef>
#include <cstdint>

namespace llm_engine {
    // Declaração da função para que o compilador saiba que ela existe
    float dot_product_q8_0(const int8_t* a, const float* b, size_t n);
}