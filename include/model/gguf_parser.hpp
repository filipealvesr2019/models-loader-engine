#pragma once
#include "../core/mapper.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace llm_engine {

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t shape[4]; // Suporte para até 4 dimensões
    uint32_t type;     // Ex: 0 = FP32, 2 = Q4_0, etc.
    size_t offset;     // Deslocamento relativo ao início dos dados (após o header)
};

class GGUFParser {
private:
    const ModelMapper& mapper;
    
public:
    explicit GGUFParser(const ModelMapper& m) : mapper(m) {}

    // Valida o header e pula para a tabela de tensores
    void parse_metadata();

    // Retorna a tabela indexada de todos os tensores
    std::unordered_map<std::string, TensorInfo> get_tensor_table();
};

}