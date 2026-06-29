#include "include/model/gguf_parser.hpp"
#include "include/core/timer.hpp"
#include <iostream>
#include <vector>
#include <fstream>

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    const std::string path = argv[1];

    // --- TESTE A: Leitura Tradicional (Copy) ---
    Timer t1;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    std::cout << "Tradicional (ifstream): " << t1.elapsed_ms() << " ms" << std::endl;

    // --- TESTE B: Mmap (Zero-Copy) ---
    Timer t2;
    GGUFParser parser;
    parser.init(path);
    auto table = parser.get_tensor_table();
    std::cout << "Mmap (Zero-Copy): " << t2.elapsed_ms() << " ms" << std::endl;

    return 0;
}