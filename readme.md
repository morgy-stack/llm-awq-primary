# llm-awq-primary

An efficient and accurate low-bit weight quantization(INT3/4) method for LLMs.

## Built with

- Python (74 files)
- C (17 files)
- Shell (16 files)
- C++ (3 files)

## Getting started

```sh
pip install .
```

## Project structure

```
llm-awq-primary/
    awq/
        kernels/
            csrc/
                attention/
                    cuda_bf16_fallbacks.cuh
                    cuda_bf16_wrapper.h
                    decoder_masked_multihead_attention.cu
                    decoder_masked_multihead_attention.h
                    decoder_masked_multihead_attention_template.hpp
                    decoder_masked_multihead_attention_utils.h
                    ft_attention.cpp
                    ft_attention.h
                    setup.py
                layernorm/
                    layernorm.cu
                    layernorm.h
                    reduction.cuh
                position_embedding/
                    pos_encoding.h
                    pos_encoding_kernels.cu
                quantization/
                    dequantize.cuh
                    gemm_cuda.h
                    gemm_cuda_gen.cu
                    gemv_cuda.cu
                    gemv_cuda.h
                quantization_new/
                    gemm/
                        gemm_cuda.cu
                        gemm_cuda.h
                        semaphore.h
                    gemv/
                        gemv_cuda.cu
                        gemv_cuda.h
                    dequantize.cuh
                    dispatch_utils.cuh
                rope_new/
                    fused_rope_with_pos.cu
                    fused_rope_with_pos.h
                w8a8/
    ...
```
