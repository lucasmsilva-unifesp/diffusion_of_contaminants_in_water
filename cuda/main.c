#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 5000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

#define THREADS_PER_BLOCK 512

__global__ void diff_eq_kernel(double *C, double *C_new, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= n && index < n * (n - 1) && index % n != 0 && index % n != n - 1) {
        C_new[index] = C[index] + D * DELTA_T * (
            (C[index+1] + C[index-1] + C[index+N] + C[index-N] - 4*C[index]) / (DELTA_X * DELTA_X)
        );
    }
}

int main(int argc, char **argv)
{
    int n_threads_per_block = THREADS_PER_BLOCK; 

    if (argc == 2) {
        n_threads_per_block = atoi(argv1[1]);
    }

    long size = N * N * sizeof(double);
    double *d_C, *d_C_new;

    cudaMalloc( (void **) &d_C, size);
    cudaMalloc( (void **) &d_C_new, size);

    double *C = (double *)malloc(size);
    double *C_new = (double *)malloc(size);

    if (!C || !C_new) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0;
    }

    memset(C, 0, size);
    memset(C_new, 0, size);

    C[(N/2) * N + (N/2)] = 1.0;

    cudaMemcpy( d_C, C, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_C_new, C_new, size, cudaMemcpyHostToDevice );

    double *temp;

    for (int t = 0; t < T; t++) {
        diff_eq<<< (N * N + (n_threads_per_block-1)) / n_threads_per_block, n_threads_per_block >>>(d_C, d_C_new, N);

        cudaDeviceSynchronize();

        temp = d_C;
        d_C = d_C_new;
        d_C_new = temp;
    }

    cudaMemcpy( C, d_C, size, cudaMemcpyDeviceToHost );

    printf("Concentração final no centro: %f\n", C_new[(N/2) * N + (N/2)]);

    free(C);
    cudaFree(d_C);
    cudaFree(d_C_new);

    return 0;
}
