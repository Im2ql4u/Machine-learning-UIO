IN3200/IN4200 Obligatory Assignment 2

This project contains two implementations of the 3D Gauss–Seidel "wavefront" algorithm:

Serial version (in folder "serial/"):
  - allocate_array3D.c
  - euclidean_distance.c
  - GS_iteration_normal.c
  - GS_iteration_2_chunks.c
  - main_serial.c

MPI version (in folder "mpi/"):
  - function_declarations_mpi.h
  - allocate_array3D_mpi.c
  - allocate_array3D.c
  - euclidean_distance.c
  - GS_iteration_2_chunks.c
  - GS_iteration_2_chunks_mpi.c
  - main_mpi.c

Serial Version:
  Compile (from serial/ folder):
    gcc main_serial.c allocate_array3D.c euclidean_distance.c GS_iteration_normal.c GS_iteration_2_chunks.c -o serial_version

  Run:
    ./serial_version <num_iters> <kmax> <jmax> <imax>
  Example:
    ./serial_version 10 20 30 40

MPI Version:
  Compile (from mpi/ folder):
    mpicc -I/opt/homebrew/include main_mpi.c allocate_array3D_mpi.c GS_iteration_2_chunks_mpi.c allocate_array3D.c GS_iteration_2_chunks.c euclidean_distance.c -o mpi_version
  
Adjust include path if mpi.h is elsewhere.

  Run with 2 processes:
    mpirun -np 2 ./mpi_version <num_iters> <kmax> <jmax> <imax>
  Example:
    mpirun -np 2 ./mpi_version 10 20 30 40
