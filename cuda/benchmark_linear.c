#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 5000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

#define THREADS_PER_BLOCK 512
#define SIZE N*N*sizeof(double)

__global__ void diff_eq_kernel(double *C, double *C_new, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= n && index < n * (n - 1) && index % n != 0 && index % n != n - 1) {
        C_new[index] = C[index] + D * DELTA_T * (
            (C[index+1] + C[index-1] + C[index+N] + C[index-N] - 4*C[index]) / (DELTA_X * DELTA_X)
        );
    }
}

double diff_eq(double *C, double *C_new) {
    double *d_C, *d_C_new;
    double *swap;

    cudaMalloc( (void **) &d_C, SIZE);
    cudaMalloc( (void **) &d_C_new, SIZE);

    cudaMemcpy(d_C, C, SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy(d_C_new, C_new, SIZE, cudaMemcpyHostToDevice );

    double difmedio;

    for (int t = 0; t < T; t++) {
        diff_eq_kernel<<< (N * N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_C, d_C_new, N);

        cudaDeviceSynchronize();

        swap = d_C;
        d_C = d_C_new;
        d_C_new = swap;

        if (t%100==0) {
            difmedio = .0;
            cudaMemcpy(C, d_C, SIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_new, d_C_new, SIZE, cudaMemcpyDeviceToHost);

            for(int i=0; i<N*N; i++)
                difmedio += fabs(C[i] - C_new[i]);

            printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
        }
    }

    cudaMemcpy(C, d_C, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_C);
    cudaFree(d_C_new);

    return C[(N/2) * N + (N/2)];
}

int main(int argc, char **argv)
{
    struct timespec start_time, end_time;

    double *C = (double *)malloc(SIZE);
    double *C_new = (double *)malloc(SIZE);

    if (!C || !C_new) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0;
    }

    memset(C, 0, SIZE);
    memset(C_new, 0, SIZE);

    C[(N/2) * N + (N/2)] = 1.0;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    double central = diff_eq(C, C_new);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    printf("Concentração final no centro: %f\n", central);
    
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("tempo: %f\n", elapsed_time);

    free(C);
    free(C_new);

    return 0;
}
