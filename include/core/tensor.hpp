#pragma once
#include <cstdint>
#include <vector>

namespace llm_engine {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
    Q8_0 = 3
};

struct TensorView {
    void* raw_data;        // Endereço alinhado
    size_t num_elements;
    DataType type;
    uint64_t shape[4];

    // Função de acesso para o kernel (Zero-Copy)
    template<typename T>
    inline T* data() const { return static_cast<T*>(raw_data); }
};

}