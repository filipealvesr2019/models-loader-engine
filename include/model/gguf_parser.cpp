#include "gguf_parser.hpp"
#include <iostream>

namespace llm_engine {
    // Leitura segura de strings formatadas no padrão GGUF
    std::string read_string(uint8_t*& ptr) {
        uint64_t len = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
        std::string str(reinterpret_cast<char*>(ptr), len);
        ptr += len;
        return str;
    }

    // Pula os blocos KV (Key-Value) para chegar na tabela de tensores
    void skip_kv_pairs(uint8_t*& ptr, uint64_t n_kv) {
        for (uint64_t i = 0; i < n_kv; ++i) {
            read_string(ptr); // Pula a chave (nome da propriedade)
            uint32_t type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
            if (type == 8) read_string(ptr); // STRING
            else if (type == 9) { // ARRAY
                uint32_t arr_type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
                uint64_t arr_len = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
                ptr += (arr_len * 8); 
            } else ptr += 8; // Tipos numéricos escalares
        }
    }
}

std::unordered_map<std::string, TensorInfo> GGUFParser::get_tensor_table() {    uint8_t* ptr = static_cast<uint8_t*>(mapper->data());
    ptr += 8; // Pula Magic Number e Version
    uint64_t n_tensors = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
    uint64_t n_kv = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;

    llm_engine::skip_kv_pairs(ptr, n_kv);
    
    std::unordered_map<std::string, TensorInfo> table;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        info.name = llm_engine::read_string(ptr);
        info.n_dims = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
        for(int d=0; d < info.n_dims; ++d) {
            info.shape[d] = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
        }
        info.type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
        info.offset = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
        table[info.name] = info;
    }
    return table;
}