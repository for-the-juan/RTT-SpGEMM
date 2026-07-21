# CUDA toolchain. CUDA_PATH is optional because nvcc provides its own include
# path when it is resolved from PATH.
CUDA_INSTALL_PATH ?=
CUDA_PATH ?= $(CUDA_INSTALL_PATH)
NVCC ?= nvcc

TARGET ?= test
GPU_PREPROCESS_TEST_TARGET ?= gpu_dmma_tiles_test
DMMA_DYNAMIC_TEST_TARGET ?= dmma_dynamic_b_test
TILE2CSR_RESTORE_TEST_TARGET ?= tile2csr_restore_test
DMMA_WIDE_TEST_TARGET ?= dmma_wide_mask_spa_test
DMMA_EXACT_MASK_HOST_TEST_TARGET ?= dmma_exact_mask_host_test

VALUE_TYPE ?= double
NVCC_FLAGS ?= -O3 -w -std=c++17
GENCODE_FLAGS ?= -gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_90,code=sm_90
INCLUDES = $(if $(strip $(CUDA_PATH)),-I$(CUDA_PATH)/include)
LIBS = $(if $(strip $(CUDA_PATH)),-L$(CUDA_PATH)/lib64) -lcudart -lcusparse
COMMON_FLAGS = $(NVCC_FLAGS) $(GENCODE_FLAGS) -Xcompiler -fopenmp \
	-Xcompiler -mfma $(INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE)

.PHONY: make gpu-preprocess-test gpu-preprocess-test-run \
	dmma-dynamic-test dmma-dynamic-test-run \
	tile2csr-restore-test tile2csr-restore-test-run \
	dmma-wide-test dmma-wide-test-run \
	dmma-exact-mask-host-test dmma-exact-mask-host-test-run

make:
	$(NVCC) $(COMMON_FLAGS) main.cu -o $(TARGET) $(LIBS) $(OPTIONS)

gpu-preprocess-test:
	$(NVCC) $(COMMON_FLAGS) ../tests/gpu_dmma_tiles_test.cu \
		-o $(GPU_PREPROCESS_TEST_TARGET) $(LIBS) $(OPTIONS)

gpu-preprocess-test-run: gpu-preprocess-test
	./$(GPU_PREPROCESS_TEST_TARGET) --device $(or $(DEVICE),0)

dmma-dynamic-test:
	$(NVCC) $(COMMON_FLAGS) ../tests/dmma_dynamic_b_test.cu \
		-o $(DMMA_DYNAMIC_TEST_TARGET) $(LIBS) $(OPTIONS)

dmma-dynamic-test-run: dmma-dynamic-test
	./$(DMMA_DYNAMIC_TEST_TARGET)

tile2csr-restore-test:
	$(NVCC) $(COMMON_FLAGS) ../tests/tile2csr_restore_test.cu \
		-o $(TILE2CSR_RESTORE_TEST_TARGET) $(LIBS) $(OPTIONS)

tile2csr-restore-test-run: tile2csr-restore-test
	./$(TILE2CSR_RESTORE_TEST_TARGET)

dmma-wide-test:
	$(NVCC) $(COMMON_FLAGS) ../tests/dmma_wide_mask_spa_test.cu \
		-o $(DMMA_WIDE_TEST_TARGET) $(LIBS) $(OPTIONS)

dmma-wide-test-run: dmma-wide-test
	./$(DMMA_WIDE_TEST_TARGET)

dmma-exact-mask-host-test:
	$(NVCC) $(COMMON_FLAGS) ../tests/dmma_exact_mask_host_test.cu \
		-o $(DMMA_EXACT_MASK_HOST_TEST_TARGET) $(LIBS) $(OPTIONS)

dmma-exact-mask-host-test-run: dmma-exact-mask-host-test
	./$(DMMA_EXACT_MASK_HOST_TEST_TARGET)
