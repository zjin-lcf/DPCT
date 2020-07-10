VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu



# Standard flags (profiling is enabled to measure device offloading time)
CFLAGS := -std=c++11 -Wall 

# Linker flags
LDFLAGS = -lm

# Codeplay Compiler
ifneq (,$(findstring codeplay,$(VENDOR)))
  CC = compute++
  CFLAGS += -no-serial-memop -sycl -sycl-driver
  LDFLAGS += -lComputeCpp
endif

# Intel DPC++ Compiler
ifneq (,$(findstring intel,$(VENDOR)))
  CC = dpcpp
  CFLAGS += 
  LDFLAGS += 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif

SRC = histogram_compare_base.cpp histogram_gmem_atomics.hpp histogram_smem_atomics.hpp test_util.hpp

EXE = histogram

$(EXE):$(SRC)
	$(CC) $(CFLAGS) $< -o $(EXE)  $(LDFLAGS) 

run: $(EXE)
	./$(EXE) --file=Austin_Evening.tga


clean: 
	rm -f $(EXE) $(EXE).linkinfo result*
