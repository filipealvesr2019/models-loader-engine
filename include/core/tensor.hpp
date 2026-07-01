#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace llm_engine {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    Q4_0 = 2,
    Q8_0 = 3,
    Q2_K = 4
};

struct TensorView {
    void* raw_data = nullptr;        // Endereço alinhado
    size_t num_elements = 0;
    DataType type = DataType::FP32;
    std::array<uint64_t, 4> shape{};

    TensorView() = default;
    TensorView(void* data, size_t elements, DataType data_type, std::array<uint64_t, 4> dims = {})
        : raw_data(data), num_elements(elements), type(data_type), shape(dims) {}

    // Função de acesso para o kernel (Zero-Copy)
    template<typename T>
    inline T* data() const { return static_cast<T*>(raw_data); }
};

}