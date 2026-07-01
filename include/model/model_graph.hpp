#pragma once
#include "../core/tensor.hpp"
#include "../layers/linear.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_engine {

class ModelGraph {
public:
    struct LayerNode {
        std::string name;
        std::function<void(const float*, float*, int, int)> forward;
        int in_dim = 0;
        int out_dim = 0;
    };

    void add_layer(const std::string& name, const LayerNode& node) {
        layers_[name] = node;
        order_.push_back(name);
    }

    const LayerNode& get_layer(const std::string& name) const {
        return layers_.at(name);
    }

    const std::vector<std::string>& order() const {
        return order_;
    }

    void run(const std::string& name, const float* input, float* output, int in_dim, int out_dim) {
        auto it = layers_.find(name);
        if (it == layers_.end()) {
            throw std::runtime_error("Layer not found: " + name);
        }
        it->second.forward(input, output, in_dim, out_dim);
    }

    void run_sequence(const std::vector<std::string>& names, const float* input, float* output, int in_dim, int out_dim) {
        std::vector<float> scratch(in_dim, 0.0f);
        std::vector<float> next(out_dim, 0.0f);
        std::copy(input, input + in_dim, scratch.begin());

        for (const auto& name : names) {
            const auto& layer = get_layer(name);
            if (layer.in_dim != static_cast<int>(scratch.size()) || layer.out_dim != static_cast<int>(next.size())) {
                next.resize(layer.out_dim, 0.0f);
                scratch.resize(layer.in_dim, 0.0f);
            }
            layer.forward(scratch.data(), next.data(), layer.in_dim, layer.out_dim);
            scratch = next;
        }

        std::copy(scratch.begin(), scratch.end(), output);
    }

private:
    std::unordered_map<std::string, LayerNode> layers_;
    std::vector<std::string> order_;
};

}