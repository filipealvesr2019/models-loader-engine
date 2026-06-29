#include "../../include/core/tensor.hpp"
#include "../../include/model/gguf_parser.hpp"

namespace llm_engine {

class TensorFactory {
public:
    static TensorView create_from_info(void* base_ptr, const TensorInfo& info) {
        TensorView tv;
        // O GGUF geralmente coloca os tensores após o header.
        // Calculamos o endereço absoluto usando a aritmética de ponteiros base.
        tv.raw_data = static_cast<uint8_t*>(base_ptr) + info.offset;
        
        // Validação de alinhamento para SIMD (64 bytes para AVX-512)
        if (reinterpret_cast<uintptr_t>(tv.raw_data) % 64 != 0) {
            // Log de aviso: em produção, o ideal é que o arquivo seja gerado 
            // já alinhado para evitar este cenário.
        }
        
        tv.type = static_cast<DataType>(info.type);
        return tv;
    }
};

}