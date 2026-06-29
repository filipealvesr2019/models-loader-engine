#include "../../include/model/gguf_parser.hpp"
#include <iostream>

namespace llm_engine {

void GGUFParser::parse_metadata() {
    uint8_t* ptr = static_cast<uint8_t*>(mapper.data());
    
    // 1. Validar Magic (GGUF)
    uint32_t magic = *reinterpret_cast<uint32_t*>(ptr);
    if (magic != 0x46554747) throw std::runtime_error("Formato GGUF inválido");

    // 2. Avançar para a versão (próximos 4 bytes)
    uint32_t version = *reinterpret_cast<uint32_t*>(ptr + 4);
    std::cout << "GGUF Versão: " << version << std::endl;

    // Nota: Em um parser real, precisaríamos iterar sobre o KV pairs
    // para encontrar o "data_offset" final onde os pesos binários começam.
}

}