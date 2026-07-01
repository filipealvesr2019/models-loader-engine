#pragma once
#include "../core/mapper.hpp"
#include <string>
#include <vector>
#include <unordered_map>

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t shape[4];
    uint32_t type;
    uint64_t offset;
};

class GGUFParser {
    MemoryMapper* mapper = nullptr;
public:
    void init(const std::string& path) { mapper = new MemoryMapper(path); }
    ~GGUFParser() { delete mapper; }
    void* data() { return mapper ? mapper->data() : nullptr; }
    std::unordered_map<std::string, TensorInfo> get_tensor_table();
};