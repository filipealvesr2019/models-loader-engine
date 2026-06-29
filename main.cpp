#include <iostream>
#include <string>
#include "model/gguf_parser.hpp" // Ajuste conforme o caminho real no seu include/

// Supondo que você terá uma classe GraphRunner para orquestrar a inferência
class GraphRunner {
public:
    void loadModel(const std::string& path) {
        std::cout << "[GraphRunner] Carregando modelo de: " << path << std::endl;
        // Aqui você chamará o seu GGUFParser
    }

    void run() {
        std::cout << "[GraphRunner] Executando grafo de tensores..." << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== LLM Engine Initializing ===" << std::endl;

    if (argc < 2) {
        std::cerr << "Uso: llm_engine <caminho_do_modelo.gguf>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];

    try {
        GraphRunner runner;
        runner.loadModel(modelPath);
        runner.run();
    } catch (const std::exception& e) {
        std::cerr << "Erro fatal: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Inference Finished ===" << std::endl;
    return 0;
}