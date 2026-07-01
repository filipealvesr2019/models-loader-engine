#include "include/model/gguf_parser.hpp"
#include "include/core/timer.hpp"
#include "include/core/tensor.hpp"
#include "src/kernels/cpu_kernels.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <windows.h>
#include <psapi.h>

static void simulate_real_load(const MemoryMapper& mapper) {
    const auto* data = static_cast<const uint8_t*>(mapper.data());
    const auto size = mapper.size();
    volatile uint8_t sum = 0;
    for (size_t i = 0; i < size; i += 4096) {
        sum += data[i];
    }
}

static void log_efficiency(const std::filesystem::path& model_path, const MemoryMapper& mapper) {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        std::cerr << "Falha ao coletar memoria do processo" << std::endl;
        return;
    }

    const auto file_size = std::filesystem::file_size(model_path);
    std::cout << "\n--- Telemetria de Eficiencia ---" << std::endl;
    std::cout << "Arquivo GGUF: " << (file_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Working Set (RAM): " << (pmc.WorkingSetSize / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Overhead relativo ao arquivo: " << (100.0 * pmc.WorkingSetSize / std::max<uint64_t>(1, file_size)) << "%" << std::endl;
}

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

            auto real_load_start = std::chrono::high_resolution_clock::now();
            simulate_real_load(*parser.mapper());
            auto real_load_end = std::chrono::high_resolution_clock::now();
            auto real_load_ms = std::chrono::duration<double, std::milli>(real_load_end - real_load_start).count();
            const auto file_size = std::filesystem::file_size(path);
            const double gb_per_s = (file_size / (1024.0 * 1024.0 * 1024.0)) / (real_load_ms / 1000.0);
            std::cout << "Carga real (touch pages): " << real_load_ms << " ms" << std::endl;
            std::cout << "Taxa de transferencia estimada: " << gb_per_s << " GB/s" << std::endl;

            log_efficiency(path, *parser.mapper());

            const size_t block_bytes = 256;
            const size_t available_bytes = std::min<size_t>(block_bytes, static_cast<size_t>(std::filesystem::file_size(path) - info.offset));
            if (available_bytes >= block_bytes) {
                std::vector<float> dequantized(256);
                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 1000; ++i) {
                    llm_engine::dequantize_block_q2_k(base + info.offset, dequantized.data(), dequantized.size());
                }
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start);

                std::cout << "Benchmark Q2_K: " << (elapsed.count() / 1000.0f) << " us / bloco" << std::endl;
                std::cout << "Primeiros valores: " << dequantized[0] << ", " << dequantized[1] << ", " << dequantized[2] << ", " << dequantized[3] << std::endl;
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << std::endl;
        return 1;
    }
}