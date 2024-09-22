


# Сетап окружения


```shell
sudo apt update
sudo apt install graphviz=2.42.2-9ubuntu0.1
python -m venv .venv
source .venv/bin/activate
pip install wheel && pip install cython # т.к. у пакета nemo-toolkit[asr] есть зависимость youtokentome, и она ожидает видеть эти пакеты установленными на момент установки себя https://github.com/VKCOM/YouTokenToMe/issues/107#issuecomment-1762142486
python -m pip install -r requirements.txt
```


Установить [nvidia-nsight](https://developer.nvidia.com/nsight-systems/get-started#Linuxx86), выполнить его требования к окружению из секции System Requirements [официального user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), например "The system’s paranoid level must be 2 or lower.



# Источники
- [Andrej Karpathy, Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Vrushank Desai, Серия постов про CUDA и устройство GPU](https://www.vrushankdes.ai/diffusion-inference-optimization)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Гайд](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) по профилировщику NVidia Nsight
- [Пост](https://pytorch.org/blog/accelerating-generative-ai-2/) от PyTorch про то, как они ускоряли языковые модели. 
- [Horace He, Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [Курс CUDA mode](https://www.youtube.com/@CUDAMODE/videos), особенно [лекция про профилирование](https://www.youtube.com/watch?v=LuhJEEJQgUM)ZZ
- Датасет речи [sova](https://github.com/sovaai/sova-dataset)

# Окружение, в котором выполнены измерения
Информация собранaная  с помощью [collect_env.py](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py)

```
PyTorch version: 2.4.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.2.0-23ubuntu4) 13.2.0
Libc version: glibc-2.39
Python version: 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-45-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3090 Ti
Nvidia driver version: 535.183.01
Is XNNPACK available: True
CPU:
    Model name:                           AMD Ryzen 9 5950X 16-Core Processor
```

Вывод `python -c 'import torch;print(*torch.__config__.show().split("\n"), sep="\n")'`

```
PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
    - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
    - Intel(R) MKL-DNN v3.4.2 (Git Hash 1137e04ec0b5251ca2b4400a4fd3c667ce843d67)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 12.1
```


"

