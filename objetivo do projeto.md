Você é um Engenheiro de Sistemas de Baixo Nível, especialista em Arquitetura de Computadores (x86_64 e ARM64), Compiladores e Álgebra Linear Computacional.



Quero desenvolver do zero, na unha e SEM usar bibliotecas de terceiros (nada de llama.cpp, ggml, onnxruntime, torch, etc.), um motor/kernel de inferência para LLMs ultra-otimizado para rodar estritamente em CPU (sem GPU ou com uso mínimo opcional), focando na redução drástica de processamento inútil (economia de 70% a 80% de CPU através de eficiência de cache e instruções nativas).



O projeto deve ser modular para suportar múltiplos formatos de modelos (Safetensors e GGUF). Quero construir isso por etapas pedagógicas e práticas. Não me dê o código do projeto inteiro de uma vez; vamos construir módulo por módulo.



Para esta nossa primeira iteração, forneça o plano de arquitetura e o código para a ETAPA 1:



---

ETAPA 1: O PARSER BINÁRIO E MEMORY MAPPING (mmap)

Quero o código limpo (especifique se prefere C, C++ ou Rust) para:

1. Abrir um arquivo de modelo (GGUF ou Safetensors) usando mapeamento de memória nativo do Sistema Operacional (mmap no Linux/macOS ou CreateFileMapping no Windows). Nada de ler o arquivo inteiro para a Heap.

2. Criar o Parser do Cabeçalho:

- Se for Safetensors: Ler os primeiros 8 bytes (tamanho do JSON), extrair a string JSON do cabeçalho e parsear os metadados (nome do tensor, shape, data_type e offsets binários).

- Se for GGUF: Validar o Magic Number ("GGUF"), ler a versão, ler a contagem de tensores/metadados e indexar a tabela de tensores (offsets, tipos de quantização e shapes).

3. Retornar uma estrutura de dados em memória que aponte exatamente para o endereço de memória virtual onde começam os pesos de cada tensor, sem ter copiado um único byte do arquivo original.



---

REQUISITOS DE CÓDIGO PARA O GEMINI:

- Escreva código idiomático de baixo nível, com tratamento de erros robusto para falhas de mmap ou arquivos corrompidos.

- Inclua comentários cirúrgicos explicando o alinhamento de memória e os structs binários.

- Explique brevemente como a paginação de memória do Sistema Operacional vai trabalhar a nosso favor aqui para manter o consumo de RAM inicial perto de zero.



Apresente a arquitetura sugerida e o código da ETAPA 1 para começarmos.

o objetivo e economizar ate 80% ou mais em processamento ao rodar modelos de varios formatos 
