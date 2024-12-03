#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 7000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new) { //diff_eq(double C[N][N], double C_new[N][N]) {
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Atualizar matriz para a próxima iteração
        double difmedio = 0.;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        if ((t%100) == 0)
            printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main() {
    struct timespec start_time, end_time;

    // Concentração inicial
    double **C = (double **)malloc(N * sizeof(double *));
    if (C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");

        return 1;
    }

    for (int i = 0; i < N; i++) {
        C[i] = (double *)malloc(N * sizeof(double));

        if (C[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");

            return 1;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.;
        }
    }

    // Concentração para a próxima iteração
    double **C_new = (double **)malloc(N * sizeof(double *));
    if (C_new == NULL) {
        fprintf(stderr, "Memory allocation failed\n");

        return 1;
    }

    for (int i = 0; i < N; i++) {
        C_new[i] = (double *)malloc(N * sizeof(double));

        if (C_new[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_new[i][j] = 0.;
        }
    }

    // Inicializar uma concentração alta no centro
    C[N/2][N/2] = 1.0;

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

    return 0;
}
