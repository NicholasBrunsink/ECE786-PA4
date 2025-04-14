NVCC := nvcc
CFLAGS := -O3

all: softmaxV1 softmaxV2

softmaxV1: softmaxV1.cu
	$(NVCC) $(CFLAGS) softmaxV1.cu -o softmaxV1

softmaxV2: softmaxV2.cu
	$(NVCC) $(CFLAGS) softmaxV2.cu -o softmaxV2

softmaxV3: softmaxV3.cu
	$(NVCC) $(CFLAGS) softmaxV3.cu -o softmaxV3

clean:
	rm -f softmaxV1 softmaxV2