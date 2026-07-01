#pragma once
#include "../core/tensor.hpp"
#include "../../src/kernels/cpu_kernels.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace llm_engine {

class LinearLayer {
private:
    TensorView weights;
    TensorView bias;

public:
    LinearLayer(TensorView w, TensorView b = {}) : weights(w), bias(b) {}

    void forward(const float* input, float* output, int in_dim, int out_dim) {
        if (!weights.raw_data || !output || !input) {
            return;
        }

        constexpr size_t kQ2KBlockSize = 256;
        std::vector<float> dequantized(kQ2KBlockSize, 0.0f);
        for (int i = 0; i < out_dim; ++i) {
            const auto* packed = static_cast<const uint8_t*>(weights.raw_data) + (i * kQ2KBlockSize);
            dequantize_block_q2_k(packed, dequantized.data(), kQ2KBlockSize);

            float acc = 0.0f;
            const int usable = std::min<int>(in_dim, static_cast<int>(kQ2KBlockSize));
            for (int j = 0; j < usable; ++j) {
                acc += input[j] * dequantized[j];
            }

            output[i] = acc + (bias.raw_data ? bias.data<float>()[i] : 0.0f);
        }
    }
};

}