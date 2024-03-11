#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char *argv[]) {
    double start_time, main_time;
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    sleep(30);
    MPI_Barrier(MPI_COMM_WORLD);
    
    main_time = MPI_Wtime() - start_time;
    if (rank == 0) printf("Time for work is %lf seconds\n", main_time);

    MPI_Finalize();
    return 0;
}