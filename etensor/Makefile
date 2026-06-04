PYTHON ?= python
NVCC ?= nvcc

BUILD_DIR := build
PREFIX ?= $(BUILD_DIR)/etensor_demo
CU ?= $(PREFIX).cu
LIB ?= $(PREFIX).so
TEST_BIN := $(BUILD_DIR)/etensor_api_test

FFI_INCLUDES = -I. $(shell $(PYTHON) -c "import tvm_ffi.libinfo as x; print(' '.join('-I' + p for p in x.include_paths()))")
FFI_LIB = $(shell $(PYTHON) -c "import tvm_ffi.libinfo as x; print(x.find_libtvm_ffi())")
FFI_LIB_DIR = $(dir $(FFI_LIB))

NVCC_FLAGS := -O3 -std=c++17 -shared -Xcompiler -fPIC -gencode arch=compute_80,code=compute_80
TEST_NVCC_FLAGS := -O3 -std=c++17 -Xcompiler -fPIC -gencode arch=compute_80,code=compute_80

.PHONY: all test clean

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $@

$(LIB): $(CU) include/etensor.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(FFI_INCLUDES) $(CU) -o $@ -L$(FFI_LIB_DIR) -ltvm_ffi -Xlinker -rpath -Xlinker $(FFI_LIB_DIR)

$(TEST_BIN): tests/etensor_api_test.cpp tests/etensor_api_test.cu include/etensor.cuh | $(BUILD_DIR)
	$(NVCC) $(TEST_NVCC_FLAGS) -I. $(filter %.cpp %.cu,$^) -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
