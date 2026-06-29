#include "../../include/model/gguf_parser.hpp"
#include <string>

namespace llm_engine {

// Função auxiliar para ler strings formatadas em GGUF
std::string read_string(uint8_t*& ptr) {
    uint64_t len = *reinterpret_cast<uint64_t*>(ptr);
    ptr += 8;
    std::string str(reinterpret_cast<char*>(ptr), len);
    ptr += len;
    return str;
}

std::unordered_map<std::string, TensorInfo> GGUFParser::get_tensor_table() {
    uint8_t* ptr = static_cast<uint8_t*>(mapper.data());
    ptr += 12; // Pula Magic (4) + Version (4) + N_KV (4)

    // 1. Pular blocos KV (o motor não precisa deles agora, só dos tensores)
    // Em um cenário real, aqui teríamos um loop que lê o tipo da KV e pula o tamanho
    
    // 2. Ler número de tensores
    uint64_t n_tensors = *reinterpret_cast<uint64_t*>(ptr);
    ptr += 8;
    
    std::unordered_map<std::string, TensorInfo> tensor_table;
    
    // 3. Iterar e indexar cada tensor
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo info;
        // Lógica de leitura seguindo a especificação GGUF:
        // Nome -> N_dims -> Dimensions[] -> Type -> Offset
        info.name = read_string(ptr);
        info.n_dims = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
        
        for(int d=0; d < info.n_dims; ++d) {
            info.shape[d] = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
        }
        
        info.type = *reinterpret_cast<uint32_t*>(ptr); ptr += 4;
        info.offset = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
        
        tensor_table[info.name] = info;
    }
    
    return tensor_table;
}

}