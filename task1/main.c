#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#define MAX_SLEEP_TIME 10
const char *critical_file = "critical.txt";

#include "mpi.h"
#include "queue.h"

#define START_ENTER 0
#define FINISH_ENTER 1
#define ALLOW_ENTER 2

#define DIM 4


int
main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	srand(time(NULL));
    MPI_Comm comm;
	int rank;
	int size;
	int fd;
    char buf;
    MPI_Status status;
    MPI_Request req;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int master_rank;
    int dims[2] = {DIM, DIM};
    int periodic[2] = {0};
    // создание транспьютерной матрицы
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &comm);
    int coords[2];
    int master_coords[2] = {0};
    MPI_Cart_coords(comm, rank, 2, coords);
    MPI_Cart_rank(comm, coords, &rank);
    MPI_Cart_rank(comm, master_coords, &master_rank);


    if (rank != master_rank) {
        MPI_Send(&buf, 1, MPI_CHAR, 0, START_ENTER, comm);
        printf("Отправлен запрос на разрешение на вход для %d\n", rank);
        MPI_Recv(&buf, 1, MPI_CHAR, 0, ALLOW_ENTER, comm, &status);
        printf("Получено разрешение на вход для %d\n", rank);
    } else { // coordinator actions
        int i;
        for (i = 0; i < 2 * SIZE; ++i) {
            MPI_Recv(&buf, 1, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
            if (status.MPI_TAG == START_ENTER) {
                if (is_free) {
                    is_free = 0;
                    printf("Разрешен вход для %d\n", status.MPI_SOURCE);
                    MPI_Isend(&buf, 1, MPI_CHAR, status.MPI_SOURCE, ALLOW_ENTER, comm, &req);
                } else {
                    enQueue(status.MPI_SOURCE);
                }
            } else if (status.MPI_TAG == FINISH_ENTER) {
                is_free = 1;
                int dst = deQueue();
                printf("Завершена обработка критической секции для %d\n", status.MPI_SOURCE);
                if (dst != -1) {
                    is_free = 0;
                    printf("Разрешен вход для %d\n", dst);
                    MPI_Isend(&buf, 1, MPI_CHAR, dst, ALLOW_ENTER, comm, &req);
                }
            }
        }
    }

// enter_critical_section
// <проверка наличия файла “critical.txt”>;
	if (!access(critical_file, F_OK)) {

		printf("Ошибка: файл %s существовал во время обработки критической секции %d\n", critical_file, rank);
		MPI_Abort(comm, MPI_ERR_FILE_EXISTS);

	} else {

		fd = open(critical_file, O_CREAT, S_IRWXU);
		if (!fd)
		{
			printf("Ошибка: невозможно создать файл %s процессом %d\n", critical_file, rank);
			MPI_Abort(comm, MPI_ERR_FILE);
		}

		int time_to_sleep = random() % MAX_SLEEP_TIME;
		sleep(time_to_sleep);

		if (close(fd))
		{
			printf("Ошибка: не удалось закрыть файл %s процессом %d\n", critical_file, rank);
			MPI_Abort(comm, MPI_ERR_FILE);
		}

		if (remove(critical_file))
		{
			printf("Ошибка: не удалось удалить файл %s процессом %d\n", critical_file, rank);
			MPI_Abort(comm, MPI_ERR_FILE);
		}
	}

    if (rank != master_rank) { // finish enter critical section
        MPI_Send(&buf, 1, MPI_CHAR, 0, FINISH_ENTER, comm);
    } else { // coordinator actions
        printf("Завершена обработка критической секции для координатора\n");
    }

    MPI_Finalize();
    return 0;
}
