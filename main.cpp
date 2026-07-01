#include "include/model/gguf_parser.hpp"
#include "include/core/timer.hpp"
#include "include/core/tensor.hpp"
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

        if (!table.empty()) {
            auto it = table.begin();
            const auto& info = it->second;
            std::array<uint64_t, 4> dims{};
            size_t num_elements = 1;
            for (uint32_t d = 0; d < info.n_dims; ++d) {
                dims[d] = info.shape[d];
                num_elements *= info.shape[d];
            }

            auto to_data_type = [](uint32_t raw_type) {
                switch (raw_type) {
                    case 0: return llm_engine::DataType::FP32;
                    case 1: return llm_engine::DataType::FP16;
                    case 2: return llm_engine::DataType::Q4_0;
                    case 3: return llm_engine::DataType::Q8_0;
                    default: return llm_engine::DataType::Q2_K;
                }
            };

            auto* base = static_cast<uint8_t*>(parser.data());
            llm_engine::TensorView tensor(base + info.offset, num_elements, to_data_type(info.type), dims);
            std::cout << "Exemplo de tensor: " << it->first << " | elementos=" << tensor.num_elements
                      << " | tipo=" << static_cast<uint32_t>(tensor.type) << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << std::endl;
        return 1;
    }
}