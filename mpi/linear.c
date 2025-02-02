#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define N 5000 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão

#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new, int local_row) {
    for (int i = 1; i < local_row - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                (C[(i+1) * N + j] + C[(i-1) * N + j] + C[i * N + (j+1)] + C[i * N + (j-1)] - 4 * C[i * N + j])  / (DELTA_X * DELTA_X)
            );
        }
    }
}

void diff_eq_mpi(double *C_local, double *C_local_new, int local_row, int myid, int numprocess, MPI_Comm comm) {
    MPI_Request requests[4];
    MPI_Status statuses[4];
    double difmedio = 0, difmedio_local = 0;

    for (int t = 0; t < T; t++) {
        if (myid > 0) {
            MPI_Isend(&C_local[N], N, MPI_DOUBLE, myid - 1, 0, comm, &requests[0]);
            MPI_Irecv(&C_local[0], N, MPI_DOUBLE, myid - 1, 1, comm, &requests[1]);
        }
        if (myid < numprocess - 1) {
            MPI_Isend(&C_local[(local_row - 2) * N], N, MPI_DOUBLE, myid + 1, 1, comm, &requests[2]);
            MPI_Irecv(&C_local[(local_row - 1) * N], N, MPI_DOUBLE, myid + 1, 0, comm, &requests[3]);
        }

        if (myid > 0)
            MPI_Waitall(2, requests, statuses);
        
        if (myid < numprocess - 1)
            MPI_Waitall(2, requests + 2, statuses);

        diff_eq(C_local, C_local_new, local_row);

        difmedio_local = 0;
        for (int i = 1; i < local_row - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio_local += fabs(C_local_new[i * N + j] - C_local[i * N + j]);
                C_local[i * N + j] = C_local_new[i * N + j];
            }
        }

        difmedio = 0;
        MPI_Allreduce(&difmedio_local, &difmedio, 1, MPI_DOUBLE, MPI_SUM, comm);

        if ((t%100 == 0) && (myid == 0)) {
            printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
        }
    }
}

int main(int argc, char **argv) {
    int myid, numprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    int local_row = N / numprocs + 2;

    double *C_local = (double *)malloc(local_row * N * sizeof(double));
    double *C_local_new = (double *)malloc(local_row * N * sizeof(double));

    if (!C_local || !C_local_new) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    memset(C_local, 0, local_row * N * sizeof(double));
    memset(C_local_new, 0, local_row * N * sizeof(double));

    if (myid == numprocs / 2)
        C_local[(local_row/2) * N + (N/2)] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);

    diff_eq_mpi(C_local, C_local_new, local_row, myid, numprocs, MPI_COMM_WORLD);

    if (myid == numprocs / 2)
        printf("Concentração inicial: %f\n", C_local[(local_row/2) * N + (N/2)]);

    free(C_local);
    free(C_local_new);

    MPI_Finalize();

    return 0;
}