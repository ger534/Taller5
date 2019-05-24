
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

// Matrix multiplication kernel called by MatAdd()
void MatAddNoKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of C

    for(int i = 0; i < C.height; i++){
        for(int j = 0; j < C.width; j++){
           C.elements[i*C.width + j] = (A.elements[i * A.width + j]) + (B.elements[i * B.width + j]);
        } 
      }
}

void MatAdd(const Matrix A, const Matrix B, Matrix C) {
    

    // Invoke no kernel
    MatAddNoKernel(A, B, C);
    
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

    MatAdd(A, B, C);

    printf("C: \n");
    for(int i = 0; i < C.height; i++){
      for(int j = 0; j < C.width; j++){
         printf("%f \t", C.elements[i*C.width + j]);
      } 
    }
    printf("\n");
}
