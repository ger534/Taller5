
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

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


void MatMul(const Matrix A, const Matrix B, Matrix C) {
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
    dim3 dimGrid(BLOCK_SIZE);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaThreadSynchronize();
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= A.height || col >= B.width) return;

    for (int e = 0; e < A.width; ++e)
        Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
    C.elements[row * C.width + col] = Cvalue;

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
         printf("%f ", A.elements[i*A.width + j]);
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
         printf("%f ", B.elements[i*B.width + j]);
      } 
    }
    printf("\n");    


    // Allocate C in device memory
    Matrix C;
    C.width = SIZE;
    C.height = SIZE;

    //memory allocation	
    C.elements = (float *) malloc(SIZE*SIZE*sizeof(float));

    MatMul(A, B, C);

    printf("C: \n");
    for(int i = 0; i < C.height; i++){
      for(int j = 0; j < C.width; j++){
         printf("%f ", C.elements[i*C.width + j]);
      } 
    }
    printf("\n");
}
