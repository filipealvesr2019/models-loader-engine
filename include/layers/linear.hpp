#pragma once
#include "../core/tensor.hpp"
#include <vector>
#include "../kernels/cpu_kernels.hpp" // Adicione esta linha!
#include <vector>
namespace llm_engine {

class LinearLayer {
private:
    TensorView weights;
    // O bias é opcional em algumas camadas, mas vamos deixar preparado
    TensorView bias; 

public:
    LinearLayer(TensorView w) : weights(w) {}

    // Executa: y = x * W^T
    // Em inferência de LLM, o formato dos tensores geralmente é [output_dim, input_dim]
   void forward(const float* input, float* output, int in_dim, int out_dim) {
            for (int i = 0; i < out_dim; ++i) {
                const int8_t* row_ptr = weights.data<int8_t>() + (i * in_dim);
                // Agora o compilador encontrará esta função
                output[i] = dot_product_q8_0(row_ptr, input, in_dim);
            }
        }
};

}