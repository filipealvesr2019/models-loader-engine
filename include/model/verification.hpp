#pragma once
#include "gguf_parser.hpp"
#include "../layers/transformer_block.hpp"
#include "../../src/kernels/cpu_kernels.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace llm_engine {

class LatencyTracker {
public:
    void begin() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double end_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_{};
};

struct VerificationResult {
    bool passed;
    double latency_ms;
};

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

    VerificationResult run_verification_and_latency(const std::string& tensor_name, const std::vector<float>& input) {
        const bool passed = run_smoke_test(tensor_name, input);

        std::vector<float> warmup_input = input;
        if (warmup_input.size() < 256) {
            warmup_input.resize(256, 1.0f);
        }

        std::vector<float> warmup_output(256, 0.0f);
        std::vector<float> output(256, 0.0f);
        auto table = parser_.get_tensor_table();
        auto it = table.find(tensor_name);
        if (it == table.end()) {
            std::cerr << "Tensor not found for latency benchmark: " << tensor_name << std::endl;
            return {passed, 0.0};
        }

        const auto& info = it->second;
        std::array<uint64_t, 4> dims{};
        size_t num_elements = 1;
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            dims[d] = info.shape[d];
            num_elements *= info.shape[d];
        }

        auto to_data_type = [](uint32_t raw_type) {
            switch (raw_type) {
                case 0: return DataType::FP32;
                case 1: return DataType::FP16;
                case 2: return DataType::Q4_0;
                case 3: return DataType::Q8_0;
                default: return DataType::Q2_K;
            }
        };

        auto* base = static_cast<uint8_t*>(parser_.data());
        TensorView tensor(base + info.offset, num_elements, to_data_type(info.type), dims);
        LinearLayer layer(tensor);
        TransformerBlock block(layer, layer);

        block.forward(warmup_input.data(), warmup_output.data(), static_cast<int>(warmup_input.size()));

        LatencyTracker tracker;
        tracker.begin();
        block.forward(warmup_input.data(), output.data(), static_cast<int>(warmup_input.size()));
        const double latency_ms = tracker.end_ms();

        std::cout << "Latencia de inferencia (warm-up + medida): " << latency_ms << " ms" << std::endl;
        return {passed, latency_ms};
    }

private:
    GGUFParser& parser_;
};

}