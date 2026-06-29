
# 1. Limpa a pasta build completamente
Remove-Item -Recurse -Force build

# 2. Reconfigura o projeto
cmake -S . -B build

# 3. Compila de novo
cmake --build build --config Release