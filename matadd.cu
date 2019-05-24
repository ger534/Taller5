
#include <stdio.h>
#include <math.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16
#define SIZE 4 

__global__ void MatAddKernel(const Matrix, const Matrix, Matrix);

void MatAdd(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x ) / dimBlock.x,
    (A.height + dimBlock.y) / dimBlock.y);
    MatAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
}

// Matrix multiplication kernel called by MatAdd()
__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    C.elements[row * C.width + col] = (A.elements[row * A.width + col]) + (B.elements[row * B.width + col]);
}

void MatAddNoKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C

    for(int i = 0; i < C.height; i++){
        for(int j = 0; j < C.width; j++){
           C.elements[i*C.width + j] = (A.elements[i * A.width + j]) + (B.elements[i * B.width + j]);
        } 
      }
}


int main(int argc, char **argv){
    printf("Begin A \n");    
   
    // Matrix 1
    Matrix A;
    A.width = SIZE;
    A.height = SIZE;
    //memory allocation	
    A.elements = (float *) malloc(SIZE*SIZE*sizeof(float));

    printf("A: \n");
    for(int i = 0; i < A.height; i++){
      for(int j = 0; j < A.width; j++){
         A.elements[i*A.width + j] = 4.0;
         printf("%f \t", A.elements[i*A.width + j]);
      } 
    }
    printf("\n");
    
    // Matrix 2
    Matrix B;
    B.width = SIZE;
    B.height = SIZE;
    //memory allocation	
    B.elements = (float *) malloc(SIZE*SIZE*sizeof(float));

    printf("B: \n");
    for(int i = 0; i < B.height; i++){
      for(int j = 0; j < B.width; j++){
         B.elements[i*B.width + j] = 4.0;
         printf("%f \t", B.elements[i*B.width + j]);
      } 
    }
    printf("\n");    


    // Allocate C in device memory
    Matrix C;
    C.width = SIZE;
    C.height = SIZE;
    //memory allocation	
    C.elements = (float *) malloc(SIZE*SIZE*sizeof(float));

    clock_t start_d=clock();
    printf("Doing GPU Vector add\n");
    MatAdd(A, B, C);

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();

    
    printf("Doing CPU Vector add\n");
    clock_t start_h = clock();
    MatAddNoKernel(A, B, C);
    clock_t end_h = clock();
	
    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    printf("GPU time = %fs \t CPU time = %fs\n", time_d, time_h);
    
    printf("C: \n");
    for(int i = 0; i < C.height; i++){
      for(int j = 0; j < C.width; j++){
         printf("%f \t", C.elements[i*C.width + j]);
      } 
    }
    printf("\n");
}
