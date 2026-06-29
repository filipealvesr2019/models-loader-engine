#include "gguf_parser.hpp"
#include <iostream>
#include <stdexcept>

namespace llm_engine {

// Leitura de string com validação de tamanho para evitar estouro de memória
std::string read_string(uint8_t*& ptr) {
    uint64_t len = *reinterpret_cast<uint64_t*>(ptr);
    ptr += 8;
    if (len > 1024) throw std::runtime_error("String de metadado muito longa!");
    std::string str(reinterpret_cast<char*>(ptr), len);
    ptr += len;
    return str;
}

// Lógica de pulo baseada no tipo exato da especificação GGUF
void skip_value(uint8_t*& ptr, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: ptr += 1; break; // U8, I8, BOOL
        case 2: case 3: ptr += 2; break;         // U16, I16
        case 4: case 5: case 6: ptr += 4; break; // U32, I32, F32
        case 10: case 11: case 12: ptr += 8; break; // U64, I64, F64
        case 8: read_string(ptr); break;         // STRING
        case 9: {                                // ARRAY
            uint32_t arr_type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
            uint64_t arr_len = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
            for (uint64_t i = 0; i < arr_len; ++i) skip_value(ptr, arr_type);
        } break;
        default: throw std::runtime_error("Tipo GGUF desconhecido!");
    }
}

void skip_kv_pairs(uint8_t*& ptr, uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; ++i) {
        read_string(ptr); // Pula a chave
        uint32_t type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
        skip_value(ptr, type);
    }
}
}

std::unordered_map<std::string, TensorInfo> GGUFParser::get_tensor_table() {
    uint8_t* ptr = static_cast<uint8_t*>(mapper->data());
    
    // Header GGUF conforme especificação:
    // Magic(4), Version(4), N_KV(8), N_Tensors(8)
    uint32_t magic = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
    uint32_t version = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
    uint64_t n_kv = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
    uint64_t n_tensors = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;

    if (magic != 0x46554747) throw std::runtime_error("Magic inválido");

    // Pula metadados KV dinâmicos
    llm_engine::skip_kv_pairs(ptr, n_kv);
    
    std::unordered_map<std::string, TensorInfo> table;
    
    // Agora o ponteiro está exatamente no início da Tabela de Tensores
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