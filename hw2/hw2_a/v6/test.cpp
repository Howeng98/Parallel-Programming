#include <iostream>
#include <emmintrin.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  // mpi argument
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double a = 2;
  __m128d a_sse;
  __m128d five;
  __m128d two;

  double checkpoint;
  double x, y;

  a_sse[0] = a_sse[1] = 2;
  five[0] = five[1] = 5;
  two[0] = two[1] = 2;
  checkpoint = MPI_Wtime();
  for (unsigned long long i = 0; i < 10000000000000; i++) {
    a = 2 * a / 5;
  }
  x = MPI_Wtime() - checkpoint;
  

  checkpoint = MPI_Wtime();
  for (unsigned long long  i = 0; i < 10000000000000; i++) {
    a_sse = _mm_mul_pd(two, _mm_mul_pd(five, a_sse));
  }
  y = MPI_Wtime() - checkpoint;

  cout << "general variable cost time: " << x << endl;
  cout << "vectorization calculation cost time: " << y << endl;

  return 0;
}