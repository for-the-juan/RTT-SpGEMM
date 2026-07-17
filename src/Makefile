# Toolchain.  With CUDA_PATH unset, use nvcc from PATH.  CUDA_INSTALL_PATH is
# retained as a compatibility alias for existing build commands.
CUDA_INSTALL_PATH ?=
CUDA_PATH ?= $(CUDA_INSTALL_PATH)
NVCC ?= nvcc
TARGET ?= test
GPU_PREPROCESS_TEST_TARGET ?= gpu_dmma_tiles_test

#GLOBAL_PARAMETERS
MAT_VAL_TYPE = double
VALUE_TYPE = double

# CUDA parameters: emit native code for both supported DMMA targets.
NVCC_FLAGS ?= -O3 -w -std=c++17
GENCODE_FLAGS ?= -gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_90,code=sm_90

# Includes. nvcc supplies its own toolkit include path when CUDA_PATH is empty.
INCLUDES = $(if $(strip $(CUDA_PATH)),-I$(CUDA_PATH)/include)

# Libraries. Preserve the existing CUDA runtime and cuSPARSE dependencies.
CUDA_LIBS = $(if $(strip $(CUDA_PATH)),-L$(CUDA_PATH)/lib64) -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o $(TARGET) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE)

.PHONY: gpu-preprocess-test gpu-preprocess-test-run

gpu-preprocess-test:
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma ../tests/gpu_dmma_tiles_test.cu -o $(GPU_PREPROCESS_TEST_TARGET) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE)

gpu-preprocess-test-run: gpu-preprocess-test
	./$(GPU_PREPROCESS_TEST_TARGET) --device $(or $(DEVICE),0)
