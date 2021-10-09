#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned long long r = atoll(argv[1]);
	unsigned long long arr[size+1];
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long begin = r / size * rank;
	unsigned long long end = r / size * rank + r / size;
	unsigned long long result = 0;
	unsigned long long sq = r*r;

	if (rank == size-1) end = r;

	if (size == 1) {
		for (unsigned long long x = begin; x < end; x+=1) {
			unsigned long long y = ceil(sqrtl(sq - x*x));
			pixels += y;
		}
		pixels %= k;
		result = pixels;
	} else {
		for (unsigned long long x = begin; x < end; x+=1) {
			unsigned long long y = ceil(sqrtl(sq - x*x));
			pixels += y;
		}
		pixels %= k;
		// MPI_Reduce(&pixels, &result, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank > 0) MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
		else if (rank == 0) {
			unsigned long long tmp;
			result += pixels;
			for (int i = 1; i < size; i++) {
				MPI_Recv(&arr[i], 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				result += arr[i];
			}
		}
	}
	MPI_Finalize();
	if (rank == 0) printf("%llu\n", (4 * result) % k);
}
