/* Include benchmark-specific header. */
#include "2mm.h"

double bench_t_start, bench_t_end;
MPI_Comm my_comm = MPI_COMM_WORLD; // current communicator
int myrank, ranksize; // current rank and number of processes in communicator


void print_array_to_file(int *a, int len_a, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < len_a; ++i) {
        fprintf(fp, "%d ", a[i]);
    }
    fclose(fp);
}

void print_array_to_file_float(float *a, int len_a, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < len_a; ++i) {
        fprintf(fp, "%lf ", a[i]);
    }
    fclose(fp);
}

void read_array_from_file(int *a, int len_a, char *filename) {
    FILE *fp = fopen(filename, "r");
    for (int i = 0; i < len_a; ++i) {
        fscanf(fp, "%d", &a[i]);
    }
    fclose(fp);
}

void read_array_from_file_float(float *a, int len_a, char *filename) {
    FILE *fp = fopen(filename, "r");
    for (int i = 0; i < len_a; ++i) {
        fscanf(fp, "%lf", &a[i]);
    }
    fclose(fp);
}

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
void MPI_kernel_2mm(int task) {
  if (task == 0) {

    printf("MPI_kernel_2mm %d\n", task);
    int step = NI / (numtasks - 2);
    int r = NI % (numtasks - 2);
    int i = 0;
    int cur = 0;

    print_array_to_file(B, NK * NJ, "B.txt");
    print_array_to_file(C, NJ * NL, "C.txt");

    printf("I printed to file\n");

    // allow calculation
    for (i = 0; i < numtasks - 2; ++i) {
      MPI_Isend(0, 0, MPI_CHAR, i + 1, ALLOW, my_comm, &status);
    }

    int tasknum;
    int idx;
    // collect results
    for (idx = 1; idx < numtasks - 1; idx++)
    {
      MPI_Recv(&tasknum, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                my_comm, &status);
      int dst = status.MPI_SOURCE;
      int start = (tasknum - 1) * step;
      int end = start + step;
      if (tasknum == PROC_NUM - 2) {
          end += r;
      }
      MPI_Recv(&D[start][0], (end - start) * NL, 
        MPI_FLOAT, dst, MPI_ANY_TAG, my_comm, &status);
      printf("Recv data from %d, %d/%d\n", status.MPI_SOURCE, idx, numtasks - 2);
      tmp_calc += 1;
    }

    printf("Message for reserve finish\n");
    MPI_Send(0, 1, MPI_CHAR, RESERVE_PROCESS, FINISH, my_comm);
    printf("Done\n");

    bench_timer_stop();
    bench_timer_print();
  } else if (task > 0) {
    printf("MPI_kernel_2mm %d\n", task);

    if (myrank == KILLED) {
      printf("****************KIIIILLLLL***********\n");
      raise(SIGKILL);
    }

    int start = (task - 1) * step;
    int end = start + step;
    if (task == PROC_NUM - 2) {
        end += r;
    }
    
    char tmp_filename_old[10];
    sprintf(tmp_filename_old, "tmp%d.txt", task);

    if (access(tmp_filename_old, F_OK) == 0) {
      read_array_from_file_float(tmp, nj * nk, tmp_filename_old);
      goto checkpoint;
    }
    
    // initialization
    printf("start row %d end row %d\n", start, end);
    for (i = start; i < end; i++)
      for (j = 0; j < nk; j++)
        A[i][j] = (float) ((i*j+1) % ni) / ni;

    read_array_from_file(&B[0][0], NK * NJ, "B.txt");
    

    for (i = start; i < end; i++)
      for (j = 0; j < nj; j++)
      {
        tmp[i][j] = 0.0f;
        for (k = 0; k < nk; ++k)
          tmp[i][j] += 1.2f * A[i][k] * B[k][j];
      }
    char tmp_filename[10];
    sprintf(tmp_filename, "tmp%d.txt", task);

    print_array_to_file_float(tmp, nj * nk, tmp_filename);

    checkpoint:

    for (i = start; i < end; i++)
      for (j = 0; j < nl; j++)
        D[i][j] = (float) (i*(j+2) % nk) / nk;

    read_array_from_file(&C[0][0], NJ * NL, "C.txt");
    for (i = start; i < end; i++)
      for (j = 0; j < nl; j++)
      {
        D[i][j] *= 1.5f;
        for (k = 0; k < nj; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
    printf("%d after calculation\n", task);
    MPI_Send(&task, 1, MPI_INT, 0, 4,
                    my_comm);
    MPI_Send(&D[start][0], (end - start) * NL, MPI_FLOAT, 0, 4,
                    my_comm);
                  
  }

}


static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {

    printf("verbose_errhandler\n");
    printf("My rank is %d\n", myrank);
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;

    MPI_Error_class(err, &eclass);

    MPI_Comm_rank(my_comm, &rank);
    MPI_Comm_size(my_comm, &size);

    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    MPI_Group_size(group_f, &nf);

    MPI_Error_string(err, errstr, &len);
    printf("Rank %d / %d:  Notified of error %s. %d found dead: ( ", 
            rank, size, errstr, nf);

            if (rank == 0) {

    ranks_gf = (int*)malloc(nf * sizeof(int));
    ranks_gc = (int*)malloc(nf * sizeof(int));

    MPI_Comm_group(comm, &group_c);

    for (i = 0; i < nf; ++i) ranks_gf[i] = i;

    MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);

    for (i = 0; i < nf; ++i) { 
      printf("%d ", ranks_gc[i]);
      if (rank == 0) {
        MPI_Send(&ranks_gc[i], 1, MPI_CHAR, RESERVE_PROCESS, ERROR, my_comm);
      } 
    }

    printf(")\n");

    // MPIX_Comm_shrink(comm, &my_comm);
    // MPI_Comm_rank(my_comm, &task);
    // MPI_Comm_size(my_comm, &ranksize);

    free(ranks_gc); free(ranks_gf);

    
        int tasknum;
        int idx;
        // collect results
        printf("PROC_NUM - 2 == %d\n", PROC_NUM - 2);
        for (idx = tmp_calc; idx < PROC_NUM - 2; idx++)
        {
          printf("%d\n", idx);
          MPI_Recv(&tasknum, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                    my_comm, &status);
          printf("GOT MSG\n");
          int dst = status.MPI_SOURCE;
          int start = (tasknum - 1) * step;
          int end = start + step;
          if (tasknum == PROC_NUM - 2) {
              end += r;
          }
          MPI_Recv(&D[start][0], (end - start) * NL, 
            MPI_FLOAT, dst, MPI_ANY_TAG, my_comm, &status);
          printf("Recv data from %d, %d/%d\n", status.MPI_SOURCE, idx, numtasks - 2);
          tmp_calc += 1;
        }

        printf("Message for reserve finish\n");
        MPI_Send(&tasknum, 1, MPI_CHAR, RESERVE_PROCESS, FINISH, my_comm);
        printf("Done\n");

        bench_timer_stop();
        bench_timer_print();
        print_array(ni, nl, D);
        MPIX_Comm_shrink(my_comm, &my_comm);
        MPI_Comm_rank(my_comm, &task);
        MPI_Comm_size(my_comm, &ranksize);
        exit(0);
    }

    MPIX_Comm_shrink(my_comm, &my_comm);

}

int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(my_comm,&myrank);
  task = myrank;
  printf("I'm %d\n", task);
  MPI_Comm_size(my_comm,&numtasks);
  step = NI / (numtasks - 2);
  r = NI % (numtasks - 2);
  
  // устанавливаем обработчик ошибок
  MPI_Errhandler errh;
  MPI_Comm_create_errhandler(verbose_errhandler, &errh);
  MPI_Comm_set_errhandler(my_comm, errh);

  MPI_Barrier(my_comm);

  if (numtasks > 1) {
    if (task == 0) {
      bench_timer_start();
      MPI_init_array();  
      MPI_kernel_2mm(task);
    } else if (task != RESERVE_PROCESS) {
      MPI_Recv(0, 0, MPI_CHAR, 0, ALLOW,
                my_comm, &status);
      MPI_kernel_2mm(task);
      // MPI_Recv(0, 0, MPI_CHAR, 0, FINISH,
      //           my_comm, &status);  
    } else {
      char buf;
      while (1) {
        MPI_Recv(&buf, 1, MPI_CHAR, 0, MPI_ANY_TAG,
                my_comm, &status);
        if (status.MPI_TAG == FINISH) {
          return;
        } else {
          printf("reserve process started\n");
          MPI_kernel_2mm(buf);
        }
      }
      
    }
  }
  if (task == 0) print_array(ni, nl, D);
  MPI_Finalize();
  return 0;
}
