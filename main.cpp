#include "include/model/gguf_parser.hpp"
#include "include/core/timer.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: llm_engine <caminho-do-modelo.gguf>" << std::endl;
        return 1;
    }

    const std::string path = argv[1];

    try {
        // --- TESTE A: Leitura Tradicional (Copy) ---
        Timer t1;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Falha ao abrir arquivo: " + path);
        }

        std::streamsize size = file.tellg();
        if (size < 0) {
            throw std::runtime_error("Falha ao ler tamanho do arquivo: " + path);
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        if (!file) {
            throw std::runtime_error("Falha ao ler o arquivo: " + path);
        }
        std::cout << "Tradicional (ifstream): " << t1.elapsed_ms() << " ms" << std::endl;

        // --- TESTE B: Mmap (Zero-Copy) ---
        Timer t2;
        GGUFParser parser;
        parser.init(path);
        auto table = parser.get_tensor_table();
        std::cout << "Mmap (Zero-Copy): " << t2.elapsed_ms() << " ms" << std::endl;
        std::cout << "Tensores lidos: " << table.size() << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << std::endl;
        return 1;
    }
}