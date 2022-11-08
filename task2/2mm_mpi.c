/* Include benchmark-specific header. */
#include "2mm.h"

double bench_t_start, bench_t_end;

static void err_handler(MPI_Comm *pcomm, int *perr, ...) {
    error_occured = 1;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int size, nf, len;
    MPI_Group group_f;

    MPI_Comm_size(main_comm, &size);
    MPIX_Comm_failure_ack(main_comm);
    MPIX_Comm_failure_get_acked(main_comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);

    ranks_gf = (int *)malloc(nf * sizeof(int));
    ranks_gc = (int *)malloc(nf * sizeof(int));
    MPI_Comm_group(comm, &group_c);
    for (i = 0; i < nf; i++)
        ranks_gf[i] = i;
    MPI_Group_translate_ranks(group_f, nf, ranks_gf,
                              group_c, ranks_gc);

    MPI_Send(0, 0, MPI_INT, RESERVE_PROCESS, ERROR, MPI_COMM_WORLD);

    free(ranks_gf);
}

void print_array_to_file(int *a, int len_a, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < len_a; ++i) {
        fprintf(fp, "%d ", a[i]);
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
    for (i = 0; i < numtasks - 2; ++i) {
      start_row[i] = cur;
      end_row[i] = cur += step;
    }
    end_row[numtasks - 2] += r;
    

    print_array_to_file(B, NK * NJ, "B.txt");
    print_array_to_file(C, NJ * NL, "C.txt");

    // allow calculation
    for (i = 0; i < numtasks - 2; ++i) {
      MPI_Isend(0, 0, MPI_CHAR, i + 1, ALLOW, MPI_COMM_WORLD, &status);
    }

    // collect results
    for (int dst = 1; dst < numtasks - 1; dst++)
    {
      MPI_Recv(&D[start_row[dst - 1]][0], (finish_row[dst - 1] - start_row[dst - 1]) * NL, MPI_FLOAT, MPI_ANY_SOURCE, dst,
                MPI_COMM_WORLD, &status);
      printf("Recv data from %d\n", status.MPI_SOURCE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(0, 0, MPI_INT, RESERVE_PROCESS, FINISH, MPI_COMM_WORLD);
    printf("Done\n");

    bench_timer_stop();
    bench_timer_print();
  } else if (task > 0) {
    printf("MPI_kernel_2mm %d\n", task);
    
    if (tmp_calc[task] == 1) {
      char tmp_filename[10];
      sprintf(tmp_filename, "tmp%d.txt", rank);
      read_array_from_file(tmp, nj * nk, tmp_filename);
      goto checkpoint;
    }
    
    // initialization
    for (i = start_row[task - 1]; i < end_row[task - 1]; i++)
      for (j = 0; j < nk; j++)
        A[i][j] = (float) ((i*j+1) % ni) / ni;

    read_array_from_file(&B[0][0], NK * NJ, "B.txt");
    

    for (i = start_row[task - 1]; i < end_row[task - 1]; i++)
      for (j = 0; j < nj; j++)
      {
        tmp[i][j] = 0.0f;
        for (k = 0; k < nk; ++k)
          tmp[i][j] += 1.2f * A[i][k] * B[k][j];
      }
    char tmp_filename[10];
    sprintf(tmp_filename, "tmp%d.txt", task);

    print_array_to_file(tmp, nj * nk, tmp_filename);
    tmp_calc[task] = 1;

    checkpoint:

    for (i = start_row[task - 1]; i < end_row[task - 1]; i++)
      for (j = 0; j < nl; j++)
        D[i][j] = (float) (i*(j+2) % nk) / nk;

    read_array_from_file(&C[0][0], NJ * NL, "C.txt");
    for (i = start_row[task - 1]; i < end_row[task - 1]; i++)
      for (j = 0; j < nl; j++)
      {
        D[i][j] *= 1.5f;
        for (k = 0; k < nj; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
    printf("%d after calculation\n", task);
    MPI_Send(&D[start_row[task - 1]][0], (end_row[task - 1] - start_row[task - 1]) * NL, MPI_FLOAT, 0, 4,
                    MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
                    
  }

}


int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&task);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  
  // устанавливаем обработчик ошибок
  MPI_Errhandler errh;
  MPI_Comm_create_errhandler(err_handler, &errh);
  MPI_Comm_set_errhandler(main_comm, errh);

  MPI_Barrier(MPI_COMM_WORLD);

  // для каждого процесса формируем имя файла для записи данных контрольных точек
  itoa(rank, filename);
  strcat(filename, ".txt");

  if (numtasks > 1) {
    if (task == 0) {
      bench_timer_start();
      MPI_init_array();  
    } else if (task == RESERVE_PROCESS) {
      MPI_Recv(0, 0, MPI_CHAR, 0, MPI_ANY_TAG,
                MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == ERROR) {
        int i;
        for (i = 0; i < nf; i++) {
          MPI_kernel_2mm(ranks_gc[i]);
        }
        MPI_Recv(0, 0, MPI_CHAR, 0, FINISH,
                MPI_COMM_WORLD, &status);
      }
      
    } else {
      MPI_Recv(0, 0, MPI_CHAR, 0, ALLOW,
                MPI_COMM_WORLD, &status);
      MPI_kernel_2mm(task);  
    }
  }
  if (task == 0) print_array(ni, nl, D);
  MPI_Finalize();
  return 0;
}
