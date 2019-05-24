NVCC = nvcc

all: matadd

%.o : %.cu
	$(NVCC) -c $< -o $@

matadd : matadd.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a matadd
