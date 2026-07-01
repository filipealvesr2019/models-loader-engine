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

        std::vector<float> dequantized(in_dim);
        for (int i = 0; i < out_dim; ++i) {
            const auto* packed = static_cast<const uint8_t*>(weights.raw_data) + (i * in_dim);
            dequantize_block_q2_k(packed, dequantized.data(), static_cast<size_t>(in_dim));

            float acc = 0.0f;
            for (int j = 0; j < in_dim; ++j) {
                acc += input[j] * dequantized[j];
            }

            output[i] = acc + (bias.raw_data ? bias.data<float>()[i] : 0.0f);
        }
    }
};

}