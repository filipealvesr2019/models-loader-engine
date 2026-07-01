#include "include/model/gguf_parser.hpp"
#include "include/core/timer.hpp"
#include "include/core/tensor.hpp"
#include "include/layers/linear.hpp"
#include "include/model/model_graph.hpp"
#include "include/model/model_loader.hpp"
#include "include/model/verification.hpp"
#include "include/layers/transformer_block.hpp"
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

            llm_engine::ModelLoader loader(parser);
            const auto blocks = loader.list_blocks();
            std::cout << "Blocos descobertos: " << blocks.size() << std::endl;
            for (const auto& block : blocks) {
                std::cout << "- " << block << std::endl;
            }

            llm_engine::VerificationHarness verifier(parser);
            const std::vector<float> dummy_input(256, 1.0f);
            const auto verification = verifier.run_verification_and_latency("blk.0.ffn_gate.weight", dummy_input);
            std::cout << "Smoke test verificado: " << verification.passed << std::endl;
            std::cout << "Latencia de inferencia: " << verification.latency_ms << " ms" << std::endl;
            std::cout << "RSS (Working Set): " << verification.rss_kb << " KB" << std::endl;

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
                std::vector<float> scalar_out(256);
                std::vector<float> simd_out(256);

                auto scalar_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10000; ++i) {
                    llm_engine::dequantize_block_q2_k_scalar(base + info.offset, scalar_out.data(), scalar_out.size());
                }
                auto scalar_end = std::chrono::high_resolution_clock::now();

                auto simd_start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 10000; ++i) {
                    llm_engine::dequantize_block_q2_k(base + info.offset, simd_out.data(), simd_out.size());
                }
                auto simd_end = std::chrono::high_resolution_clock::now();

                const auto scalar_us = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start).count() / 10000.0f;
                const auto simd_us = std::chrono::duration_cast<std::chrono::microseconds>(simd_end - simd_start).count() / 10000.0f;
                const float gain = 100.0f * (1.0f - (simd_us / scalar_us));

                    std::cout << "Benchmark Q2_K scalar: " << scalar_us << " us / bloco" << std::endl;
                std::cout << "Benchmark Q2_K AVX2: " << simd_us << " us / bloco" << std::endl;
                std::cout << "Ganho estimado AVX2: " << gain << "%" << std::endl;
                std::cout << "Primeiros valores: " << simd_out[0] << ", " << simd_out[1] << ", " << simd_out[2] << ", " << simd_out[3] << std::endl;

                std::vector<float> input(256, 1.0f);
                std::vector<float> output(16, 0.0f);
                llm_engine::LinearLayer layer(tensor);
                llm_engine::ModelGraph graph;
                graph.add_layer("demo_linear", {
                    "demo_linear",
                    [&layer](const float* in, float* out, int in_dim, int out_dim) {
                        layer.forward(in, out, in_dim, out_dim);
                    },
                    256,
                    16
                });
                graph.add_layer("demo_linear_2", {
                    "demo_linear_2",
                    [&layer](const float* in, float* out, int in_dim, int out_dim) {
                        layer.forward(in, out, in_dim, out_dim);
                    },
                    16,
                    16
                });

                auto forward_start = std::chrono::high_resolution_clock::now();
                std::vector<std::string> sequence = { "demo_linear", "demo_linear_2" };
                graph.run_sequence(sequence, input.data(), output.data(), 256, 16);
                auto forward_end = std::chrono::high_resolution_clock::now();
                const auto forward_us = std::chrono::duration_cast<std::chrono::microseconds>(forward_end - forward_start).count();
                std::cout << "Forward de uma camada pequena: " << forward_us << " us" << std::endl;
                std::cout << "Primeiro output: " << output[0] << std::endl;

                std::vector<float> normalized(256, 0.0f);
                std::vector<float> residual(256, 0.0f);
                llm_engine::rms_norm(input.data(), normalized.data(), input.size());
                llm_engine::add_vectors(input.data(), normalized.data(), residual.data(), input.size());
                std::cout << "RMSNorm primeiro valor: " << normalized[0] << std::endl;
                std::cout << "Residual primeiro valor: " << residual[0] << std::endl;

                llm_engine::LinearLayer attn_layer(tensor);
                llm_engine::LinearLayer mlp_layer(tensor);
                llm_engine::TransformerBlock block(attn_layer, mlp_layer);
                std::vector<float> block_output(256, 0.0f);
                auto block_start = std::chrono::high_resolution_clock::now();
                block.forward(input.data(), block_output.data(), 256);
                auto block_end = std::chrono::high_resolution_clock::now();
                const auto block_us = std::chrono::duration_cast<std::chrono::microseconds>(block_end - block_start).count();
                std::cout << "Transformer block: " << block_us << " us" << std::endl;
                std::cout << "Primeiro output do bloco: " << block_output[0] << std::endl;
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Erro: " << ex.what() << std::endl;
        return 1;
    }
}