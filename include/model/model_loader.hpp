#pragma once
#include "gguf_parser.hpp"
#include "model_graph.hpp"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace llm_engine {

class ModelLoader {
public:
    explicit ModelLoader(GGUFParser& parser) : parser_(parser) {}

    std::vector<std::string> list_blocks() const {
        const auto table = parser_.get_tensor_table();
        std::vector<std::string> blocks;
        for (const auto& [name, info] : table) {
            if (name.find("blk.") == 0) {
                const auto dot = name.find('.', 4);
                if (dot != std::string::npos) {
                    const std::string block_name = name.substr(0, dot);
                    if (std::find(blocks.begin(), blocks.end(), block_name) == blocks.end()) {
                        blocks.push_back(block_name);
                    }
                }
            }
        }
        std::sort(blocks.begin(), blocks.end());
        return blocks;
    }

    std::vector<std::string> list_tensors_for_block(const std::string& block_name) const {
        const auto table = parser_.get_tensor_table();
        std::vector<std::string> matches;
        for (const auto& [name, info] : table) {
            if (name.rfind(block_name + ".", 0) == 0) {
                matches.push_back(name);
            }
        }
        std::sort(matches.begin(), matches.end());
        return matches;
    }

private:
    GGUFParser& parser_;
};

}