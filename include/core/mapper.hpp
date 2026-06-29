#pragma once

#include <windows.h>
#include <string>
#include <stdexcept>
#include <iostream>

namespace llm_engine {

/**
 * @brief Gerenciador de memória que utiliza mmap (via Windows API)
 * para mapear arquivos de modelos diretamente no espaço de endereçamento.
 */
class ModelMapper {
private:
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
    void* lpBaseAddress = nullptr;
    size_t fileSize = 0;

public:
    explicit ModelMapper(const std::wstring& path) {
        // Abre o arquivo com acesso de leitura
        hFile = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, 
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Mapper: Falha ao abrir o arquivo.");
        }

        LARGE_INTEGER size;
        GetFileSizeEx(hFile, &size);
        fileSize = static_cast<size_t>(size.QuadPart);

        // Cria o objeto de mapeamento de arquivo
        hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            CloseHandle(hFile);
            throw std::runtime_error("Mapper: Falha ao criar FileMapping.");
        }

        // Mapeia o view do arquivo
        lpBaseAddress = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!lpBaseAddress) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            throw std::runtime_error("Mapper: Falha ao mapear o arquivo na memória.");
        }
    }

    ~ModelMapper() {
        if (lpBaseAddress) UnmapViewOfFile(lpBaseAddress);
        if (hMap) CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
    }

    // Proíbe cópias para evitar double-free ou erros de handle
    ModelMapper(const ModelMapper&) = delete;
    ModelMapper& operator=(const ModelMapper&) = delete;

    inline void* data() const { return lpBaseAddress; }
    inline size_t size() const { return fileSize; }
};

} // namespace llm_engine