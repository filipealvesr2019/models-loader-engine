#include "include/model/gguf_parser.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    
    try {
        GGUFParser parser;
        parser.init(argv[1]);
        auto table = parser.get_tensor_table();
        
        std::cout << "Sucesso! Total de tensores: " << table.size() << std::endl;
        for(auto const& [name, info] : table) {
            std::cout << "Tensor: " << name << " | Offset: " << info.offset << std::endl;
        }
    } catch(const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
    }
    return 0;
}