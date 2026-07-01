#include "gguf_parser.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llm_engine {

template <typename T>
T read_value(uint8_t*& ptr) {
    T value{};
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

std::string read_string(uint8_t*& ptr) {
    const uint64_t len = read_value<uint64_t>(ptr);
    if (len > 1024 * 1024) {
        throw std::runtime_error("Parser desalinhado: tamanho de string absurdo detectado.");
    }

    std::string str(reinterpret_cast<char*>(ptr), len);
    ptr += len;
    return str;
}

void skip_value(uint8_t*& ptr, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: ptr += 1; break;
        case 2: case 3: ptr += 2; break;
        case 4: case 5: case 6: ptr += 4; break;
        case 10: case 11: case 12: ptr += 8; break;
        case 8: read_string(ptr); break;
        case 9: {
            const uint32_t arr_type = read_value<uint32_t>(ptr);
            const uint64_t arr_len = read_value<uint64_t>(ptr);
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_value(ptr, arr_type);
            }
        } break;
        default: throw std::runtime_error("Tipo GGUF inválido detectado!");
    }
}

} // namespace llm_engine

std::unordered_map<std::string, TensorInfo> GGUFParser::get_tensor_table() {
    uint8_t* ptr = static_cast<uint8_t*>(mapper_->data());

    const uint32_t magic = llm_engine::read_value<uint32_t>(ptr);
    if (magic != 0x46554747) {
        throw std::runtime_error("Magic number inválido.");
    }

    const uint32_t version = llm_engine::read_value<uint32_t>(ptr);
    const uint64_t n_tensors = llm_engine::read_value<uint64_t>(ptr);
    const uint64_t n_kv = llm_engine::read_value<uint64_t>(ptr);

    std::cout << "GGUF version=" << version << " n_tensors=" << n_tensors << " n_kv=" << n_kv << std::endl;

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = llm_engine::read_string(ptr);
        const uint32_t type = llm_engine::read_value<uint32_t>(ptr);
        std::cout << "KV [" << i << "]: " << key << std::endl;
        llm_engine::skip_value(ptr, type);
    }

    std::unordered_map<std::string, TensorInfo> table;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        info.name = llm_engine::read_string(ptr);
        info.n_dims = llm_engine::read_value<uint32_t>(ptr);

        for (uint32_t d = 0; d < info.n_dims; ++d) {
            info.shape[d] = llm_engine::read_value<uint64_t>(ptr);
        }

        info.type = llm_engine::read_value<uint32_t>(ptr);
        info.offset = llm_engine::read_value<uint64_t>(ptr);

        table[info.name] = info;
    }

    return table;
}