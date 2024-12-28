#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


#define N 2000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define SIZE N * N
#define MEM_SIZE_MATRIX SIZE * sizeof(double)


__global__ void diff_eq_kernel(double *C, double *C_new, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * n + x;

    if (x > 0 && x < n - 1 && y > 0 && y < n - 1) {
        C_new[index] = C[index] + D * DELTA_T * (
            (C[index + 1] + C[index - 1] + C[index + n] + C[index - n] - 4 * C[index]) / (DELTA_X * DELTA_X)
        );
    }
}

void initialize_matrix(double *m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = 0.0;
        }
    }
    m[(n / 2) * n + (n / 2)] = 1.0; // Inicializar uma concentração alta no centro
}

void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    double *C, *C_new;
    double *d_C, *d_C_new;
    size_t size = MEM_SIZE_MATRIX;

    // Alocar memória no host
    C = (double*)malloc(size);
    C_new = (double*)malloc(size);

    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Inicializar matrizes
    initialize_matrix(C, N);
    initialize_matrix(C_new, N);

    // Alocar memória na GPU
    check_cuda_error(cudaMalloc((void**)&d_C, size), "Failed to allocate device memory for C");
    check_cuda_error(cudaMalloc((void**)&d_C_new, size), "Failed to allocate device memory for C_new");

    // Copiar dados do host para a GPU
    check_cuda_error(cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice), "Failed to copy data from host to device");

    // Configurar grade e blocos
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    double difmedio;

    // Executar kernel
    for (int t = 0; t < T; t++) {
        diff_eq_kernel<<<grid, block>>>(d_C, d_C_new, N);
        check_cuda_error(cudaDeviceSynchronize(), "Kernel execution failed");

        // Trocar ponteiros
        double *swapAux = d_C;
        d_C = d_C_new;
        d_C_new = swapAux;

        if (t % 100 == 0) {
            difmedio = 0.0;
            check_cuda_error(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");
            check_cuda_error(cudaMemcpy(C_new, d_C_new, size, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

            for (int i = 0; i < N * N; i++)
                difmedio += fabs(C[i] - C_new[i]);

            printf("interacao %d - diferenca=%g\n", t, difmedio / ((N - 2) * (N - 2)));
        }
    }

    // Copiar resultado de volta para o host
    check_cuda_error(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

    // Liberar memória na GPU
    cudaFree(d_C);
    cudaFree(d_C_new);

    // Liberar memória no host
    free(C);
    free(C_new);

    return 0;
}