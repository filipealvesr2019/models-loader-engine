cmake_minimum_required(VERSION 3.15)
project(llm_engine CXX)

set(CMAKE_CXX_STANDARD 20)

# Compila todos os .cpp que você colocar na pasta src
file(GLOB_RECURSE SOURCES "src/*.cpp" "main.cpp")

add_executable(llm_engine 
    main.cpp
    include/model/gguf_parser.cpp
    # Se você criar o timer como classe separada, adicione-o aqui
)
# O compilador olha a pasta include para achar os seus .hpp
target_include_directories(llm_engine PUBLIC include)

if(MSVC)
    target_compile_options(llm_engine PRIVATE /O2 /arch:AVX2 /EHsc)
endif()