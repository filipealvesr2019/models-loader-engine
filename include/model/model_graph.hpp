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

private:
    std::unordered_map<std::string, LayerNode> layers_;
    std::vector<std::string> order_;
};

}