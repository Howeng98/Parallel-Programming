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
  unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	unsigned long long sq = r*r;
  unsigned long long squre_long = ceil(r / sqrtl(2));
  unsigned long long squre_size = (squre_long * squre_long) % k;
  unsigned long long begin = (squre_long / size) * rank;
  unsigned long long end = begin + squre_long / size;
  unsigned long long pixels = 0;
  unsigned long long result = 0;
  unsigned long long a = ceil(sqrtl(sq-begin*begin));
  unsigned long long b = (a-1)*(a-1);
  unsigned long long add = 2 * (a-squre_long);

  if (rank == size - 1) end = squre_long;

  for (unsigned long long  x = begin; x < end; x++) {
    if (sq - x*x <= b) {
      a -= 1;
      b = (a-1)*(a-1);
      add = 2*(a-squre_long);
    }
    pixels += add;
  }
  pixels %= k;
  if (size > 0) MPI_Reduce(&pixels, &result, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  else result = pixels;
	MPI_Finalize();
  if (rank == 0) {
    result *= 4;
    result += squre_size * 4;
    result %= k;
    printf("%llu\n", result);
  }
}
