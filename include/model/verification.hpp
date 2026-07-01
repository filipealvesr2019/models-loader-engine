#pragma once
#include "gguf_parser.hpp"
#include "model_graph.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace llm_engine {

class VerificationHarness {
public:
    explicit VerificationHarness(GGUFParser& parser) : parser_(parser) {}

    bool run_smoke_test(const std::string& tensor_name, const std::vector<float>& input) {
        auto table = parser_.get_tensor_table();
        auto it = table.find(tensor_name);
        if (it == table.end()) {
            std::cerr << "Tensor not found: " << tensor_name << std::endl;
            return false;
        }

        const auto& info = it->second;
        std::vector<float> output(16, 0.0f);
        std::vector<float> input_copy = input;

        if (input_copy.size() < 256) {
            input_copy.resize(256, 1.0f);
        }

        std::vector<float> dequantized(256, 0.0f);
        const auto* base = static_cast<const uint8_t*>(parser_.data());
        dequantize_block_q2_k(base + info.offset, dequantized.data(), dequantized.size());

        float sum = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = dequantized[i % 256] * input_copy[i % 256];
            sum += output[i];
        }

        const float expected_sum = 3072.91f;
        const bool ok = std::fabs(sum - expected_sum) < 0.001f;
        std::cout << "Verification for " << tensor_name << ": " << sum << " | ok=" << ok << std::endl;
        return ok;
    }

private:
    GGUFParser& parser_;
};

}