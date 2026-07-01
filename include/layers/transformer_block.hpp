#pragma once
#include "linear.hpp"
#include <vector>

namespace llm_engine {

class TransformerBlock {
public:
    TransformerBlock(LinearLayer attn, LinearLayer mlp)
        : attn_layer_(attn), mlp_layer_(mlp) {}

    void forward(const float* input, float* output, int dim) {
        std::vector<float> normalized(dim);
        std::vector<float> attn_out(dim);
        std::vector<float> residual(dim);
        std::vector<float> mlp_out(dim);

        rms_norm(input, normalized.data(), static_cast<size_t>(dim));
        attn_layer_.forward(normalized.data(), attn_out.data(), dim, dim);
        add_vectors(input, attn_out.data(), residual.data(), static_cast<size_t>(dim));

        std::vector<float> normalized_residual(dim);
        rms_norm(residual.data(), normalized_residual.data(), static_cast<size_t>(dim));
        mlp_layer_.forward(normalized_residual.data(), mlp_out.data(), dim, dim);
        add_vectors(residual.data(), mlp_out.data(), output, static_cast<size_t>(dim));
    }

private:
    LinearLayer attn_layer_;
    LinearLayer mlp_layer_;
};

}