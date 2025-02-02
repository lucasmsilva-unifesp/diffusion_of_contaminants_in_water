#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define N 7000
#define T 500
#define D 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new, int local_rows, int start_row) {
    for (int i = 1; i < local_rows - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C_new[i][j] = C[i][j] + D * DELTA_T * (
                (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
            );
        }
    }
}

void diff_eq_mpi(double **C_local, double **C_local_new, int local_rows, int myid, int numprocs, MPI_Comm comm) {
    MPI_Request requests[4];
    MPI_Status statuses[4];
    double difmedio = 0, difmedio_local = 0;

    for (int t = 0; t < T; t++) {
        // Comunicação das linhas de halo (boundaries)
        if (myid > 0) {
            MPI_Isend(C_local[1], N, MPI_DOUBLE, myid - 1, 0, comm, &requests[0]);
            MPI_Irecv(C_local[0], N, MPI_DOUBLE, myid - 1, 1, comm, &requests[1]);
        }
        if (myid < numprocs - 1) {
            MPI_Isend(C_local[local_rows - 2], N, MPI_DOUBLE, myid + 1, 1, comm, &requests[2]);
            MPI_Irecv(C_local[local_rows - 1], N, MPI_DOUBLE, myid + 1, 0, comm, &requests[3]);
        }

        // Espera comunicação assíncrona
        if (myid > 0) MPI_Waitall(2, requests, statuses);
        if (myid < numprocs - 1) MPI_Waitall(2, &requests[2], statuses);

        // Computação da equação de difusão
        diff_eq(C_local, C_local_new, local_rows, myid * (local_rows - 2));

        // Atualização da matriz e cálculo da diferença média
        difmedio_local = 0;
        for (int i = 1; i < local_rows - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio_local += fabs(C_local_new[i][j] - C_local[i][j]);
                C_local[i][j] = C_local_new[i][j];
            }
        }

        MPI_Allreduce(&difmedio_local, &difmedio, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (myid == 0 && t % 100 == 0) {
            printf("Iteração %d - Diferença média: %g\n", t, difmedio / ((N-2)*(N-2)));
        }
    }
}

int main(int argc, char **argv) {

    int myid, numprocs;

    // Inicialização do MPI, com base no número de processos que serão inseridos na execução do programa. Também é possível dar um nome para o processo.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    struct timespec start_time, end_time;

    int local_rows = (N / numprocs) + 2; // Linhas locais + halo

    // Alocação de matrizes locais (usando array de ponteiros)
    double **C_local = (double **)malloc(local_rows * sizeof(double *));
    double **C_local_new = (double **)malloc(local_rows * sizeof(double *));

    if(C_local == NULL || C_local_new == NULL) {
        printf("Ponteiro de ponteiro: erro de alocacao\n");
        return 1;
    }

    for (int i = 0; i < local_rows; i++) {
        C_local[i] = (double *)malloc(N * sizeof(double));
        C_local_new[i] = (double *)malloc(N * sizeof(double));

        if(C_local[i] == NULL || C_local_new[i] == NULL) {
            printf("Ponteiro: erro de alocacao\n");
            return 1;
        }

        memset(C_local[i], 0, N * sizeof(double));
        memset(C_local_new[i], 0, N * sizeof(double));
    }

    // Inicialização (fonte no centro)
    if (myid == numprocs / 2) {
        int mid_row = local_rows / 2;
        C_local[mid_row][N/2] = 1.0;
    }

    // Sincronização entre processos
    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    diff_eq_mpi(C_local, C_local_new, local_rows, myid, numprocs, MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calcula o tempo em segundos
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("tempo de execução: %f\n", elapsed_time);

    // Liberação de memória
    for (int i = 0; i < local_rows; i++) {
        free(C_local[i]);
        free(C_local_new[i]);
    }
    free(C_local);
    free(C_local_new);

    MPI_Finalize();
    return 0;
}