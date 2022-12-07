/* Include benchmark-specific header. */
#include "2mm.h"

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{ 
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static
void init_array()
{
  
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (float) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (float) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (float) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (float) (i*(j+2) % nk) / nk;
}

static
void MPI_init_array()
{
    for (i = 0; i < nk; i++)
      for (j = 0; j < nj; j++)
        B[i][j] = (float) (i*(j+1) % nj) / nj;
    for (i = 0; i < nj; i++)
      for (j = 0; j < nl; j++)
        C[i][j] = (float) ((i*(j+3)+1) % nl) / nl;
}


static
void print_array(int ni, int nl,
   float D[ ni][nl])
{
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%0.2f ", D[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "D");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_2mm()
{
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      tmp[i][j] = 0.0f;
      for (k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
    {
      D[i][j] *= beta;
      for (k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }

}


static
void MPI_kernel_2mm() {
  if (task == 0) {

    printf("MPI_kernel_2mm %d\n", task);
    // send A, B, C, D
    for (int dst = 1; dst < numtasks; dst++) {
      MPI_Send(&B[0][0], NK * NJ, MPI_FLOAT, dst, 1, MPI_COMM_WORLD);
      MPI_Send(&C[0][0], NJ * NL, MPI_FLOAT, dst, 2, MPI_COMM_WORLD);
      printf("Send data to %d\n", dst);
    }

    // collect results
    for (int dst = 1; dst < numtasks; dst++)
    {
      MPI_Recv(offset, 2, MPI_INT, dst, 3, MPI_COMM_WORLD, &status);
      MPI_Recv(&D[offset[0]][0], (offset[1] - offset[0]) * NL, MPI_FLOAT, dst, 4,
                MPI_COMM_WORLD, &status);
      printf("Recv data from %d\n", dst);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Done\n");

    bench_timer_stop();
    bench_timer_print();

  } else if (task > 0) {
    printf("MPI_kernel_2mm %d\n", task);
    printf("%d has %d\n", task, offset[1] - offset[0]);
    // initialization
    for (i = offset[0]; i < offset[1]; i++)
      for (j = 0; j < nk; j++)
        A[i][j] = (float) ((i*j+1) % ni) / ni;
    for (i = offset[0]; i < offset[1]; i++)
      for (j = 0; j < nl; j++)
        D[i][j] = (float) (i*(j+2) % nk) / nk;
    MPI_Recv(&B[0][0], NK * NJ, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&C[0][0], NJ * NL, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
    for (i = offset[0]; i < offset[1]; i++)
      for (j = 0; j < nj; j++)
      {
        tmp[i][j] = 0.0f;
        for (k = 0; k < nk; ++k)
          tmp[i][j] += 1.2f * A[i][k] * B[k][j];
      }
    for (i = offset[0]; i < offset[1]; i++)
      for (j = 0; j < nl; j++)
      {
        D[i][j] *= 1.5f;
        for (k = 0; k < nj; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
    printf("%d after calculation\n", task);
    MPI_Send(offset, 2, MPI_INT, 0, 3, MPI_COMM_WORLD);
    MPI_Send(&D[offset[0]][0], (offset[1] - offset[0]) * NL, MPI_FLOAT, 0, 4,
                    MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
                    
  }

}


int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&task);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

  if (numtasks > 1) {
    if (task == 0) {
      bench_timer_start();
      MPI_init_array();  
    }

    int step = NI / (numtasks - 1);
    int r = NI % (numtasks - 1);
    if (task > 0) {
      offset[0] = step * (task - 1);
      offset[1] = (task == (numtasks - 1)) ? offset[0] + step + r : offset[0] + step;
    }
    
    MPI_kernel_2mm();  
  } else {
    if (task == 0) {
      bench_timer_start();
      init_array();
      kernel_2mm();
      bench_timer_stop();
      bench_timer_print();
    }
    
  }

  if (task == 0) print_array(ni, nl, D);
  MPI_Finalize();
  return 0;
}
