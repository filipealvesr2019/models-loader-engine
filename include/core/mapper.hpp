#pragma once
#include <windows.h>
#include <string>
#include <stdexcept>

class MemoryMapper {
    HANDLE hFile, hMap;
    void* data_ptr;
public:
    MemoryMapper(const std::string& path) {
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Falha ao abrir arquivo: " + path);
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize)) {
            CloseHandle(hFile);
            throw std::runtime_error("Falha ao obter tamanho do arquivo: " + path);
        }

        hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            CloseHandle(hFile);
            throw std::runtime_error("Falha ao mapear arquivo: " + path);
        }

        data_ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data_ptr) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            throw std::runtime_error("Falha ao criar view do arquivo: " + path);
        }
    }
    ~MemoryMapper() {
        if (data_ptr) UnmapViewOfFile(data_ptr);
        if (hMap) CloseHandle(hMap);
        if (hFile) CloseHandle(hFile);
    }
    void* data() { return data_ptr; }
};