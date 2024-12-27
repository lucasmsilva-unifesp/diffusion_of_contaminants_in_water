#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 7000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new, int n_threads) {
    omp_set_num_threads(n_threads);
    double delta_x_squared_inv = 1.0 / (DELTA_X * DELTA_X);

    for (int t = 0; t < T; t++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                    (C[(i+1) * N + j] + C[(i-1) * N + j] + C[i * N + (j+1)] + C[i * N + (j-1)] - 4 * C[i * N + j]) 
                    * delta_x_squared_inv
                );
            }
        }

        double difmedio = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:difmedio) schedule(static)
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
    double start, end;
    int n_threads;

    if (argc != 2) {
        fprintf(stderr, "use: ./main.exe <n_threads>\n");
        return 1;
    }

    n_threads = atoi(argv[1]);

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

    // começa a contagem do tempo
    start = omp_get_wtime();

    // Executar as iterações no tempo para a equação de difusão
    diff_eq(C, C_new, n_threads);

    // Fim da contagem
    end = omp_get_wtime();

    printf("%d;%f\n", n_threads, end - start);

    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", C[N/2][N/2]);    

    // Liberação de memória
    free(C);
    free(C_new);

    return 0;
}

