find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(lib STATIC
        ase.cu
        ase.h
        ase.cuh
    )
endif()
