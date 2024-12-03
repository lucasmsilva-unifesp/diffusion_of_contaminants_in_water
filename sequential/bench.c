#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define N 7000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new) {
    double delta_x_squared_inv = 1.0 / (DELTA_X * DELTA_X);

    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                    (C[(i+1) * N + j] + C[(i-1) * N + j] + C[i * N + (j+1)] + C[i * N + (j-1)] - 4 * C[i * N + j]) 
                    * delta_x_squared_inv
                );
            }
        }

        double difmedio = 0.0;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i * N + j] - C[i * N + j]);
                C[i * N + j] = C_new[i * N + j];
            }
        }

        if ((t%100) == 0)
            printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main(int argc, char **argv) {
    struct timespec start_time, end_time;

    // Alocação dinâmica para as matrizes
    double *C = (double *)malloc(N * N * sizeof(double));
    double *C_new = (double *)malloc(N * N * sizeof(double));

    if (!C || !C_new) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Inicialização
    memset(C, 0, N * N * sizeof(double));
    memset(C_new, 0, N * N * sizeof(double));

    // Inicializar uma concentração alta no centro
    C[(N/2) * N + (N/2)] = 1.0;

    // Inicia o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Executar as iterações no tempo para a equação de difusão
    diff_eq(C, C_new);

    // Para o cronômetro
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calcula o tempo em segundos
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("1;%f\n", elapsed_time);

    // Liberação de memória
    free(C);
    free(C_new);

    return 0;
}
