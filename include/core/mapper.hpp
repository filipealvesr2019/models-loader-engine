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
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Falha ao abrir arquivo");
        hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) throw std::runtime_error("Falha ao mapear arquivo");
        data_ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    }
    ~MemoryMapper() {
        if (data_ptr) UnmapViewOfFile(data_ptr);
        if (hMap) CloseHandle(hMap);
        if (hFile) CloseHandle(hFile);
    }
    void* data() { return data_ptr; }
};