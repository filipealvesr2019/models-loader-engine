#include "../../include/core/tensor.hpp"
#include <immintrin.h> // Intrínsecas para AVX2 (x86_64)

namespace llm_engine {

/**
 * @brief Dot Product otimizado para Q8_0 (Int8 * Float32)
 * @param a Pesos (quantizados em int8)
 * @param b Ativações (float32)
 * @param n Tamanho do vetor (deve ser múltiplo de 8 para AVX2)
 */
float dot_product_q8_0(const int8_t* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    
    // Processamos 8 elementos por vez usando AVX2
    for (size_t i = 0; i < n; i += 8) {
        // Carrega 8 floats de ativação
        __m256 va = _mm256_loadu_ps(&b[i]);
        
        // Converte 8 inteiros para floats (simulando a desquantização in-register)
        __m256i vi8 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&a[i]));
        __m256 vb = _mm256_cvtepi32_ps(vi8);
        
        // FMA: Fused Multiply-Add (a * b + c) - A instrução mais importante da CPU
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Redução horizontal: soma os 8 elementos do registrador YMM
    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[1] + result[2] + result[3] + 
           result[4] + result[5] + result[6] + result[7];
}

}