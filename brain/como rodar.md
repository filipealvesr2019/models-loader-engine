
# 1. Limpa a pasta build completamente
Remove-Item -Recurse -Force build

# 2. Reconfigura o projeto
cmake -S . -B build

# 3. Compila de novo
cmake --build build --config Release

# rodar modelo de IA F:\models-loader-engine>
.\build\Release\llm_engine.exe "models\seu-modelo.gguf"