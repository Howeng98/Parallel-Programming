#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int m_rank, m_size, o_rank, o_size;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &m_size);

	unsigned long long m_begin, m_end, m_total;
	unsigned long long sq = r*r;
	unsigned long long squre_long = ceil(r / sqrtl(2));
	unsigned long long squre_size = (squre_long * squre_long) % k;
	unsigned long long result = 0;
	unsigned long long ans;

	m_begin = (squre_long / m_size) * m_rank;
	m_end = (m_rank == m_size-1) ? squre_long : m_begin + squre_long / m_size;
	m_total = m_end - m_begin;

	#pragma omp parallel shared(r, k, m_begin, m_end, sq, squre_size, m_total)
	{
		o_size = omp_get_num_threads();
		o_rank = omp_get_thread_num();
		unsigned long long begin, end, a, b, add, pixels;

		pixels = 0;
		begin = m_begin + (m_total / o_size) * o_rank;
		end = (o_rank == o_size-1) ? m_end : begin + m_total / o_size;
		a = ceil(sqrtl(sq-begin*begin));
		b = (a-1)*(a-1);
		add = 2 * (a - squre_long);

		for (unsigned long long x = begin; x < end; x++) {
			if (sq - x * x <= b) {
				a -= 1;
				b = (a-1) * (a-1);
				add = 2 * (a - squre_long); 
			}
			pixels += add;
		}
		pixels %= k;

		#pragma omp critical
		{
			result += pixels;
			result %= k;
		}
	}

	MPI_Reduce(&result, &ans, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (m_rank == 0) {
		ans %= k;
		ans = (ans + squre_size) % k;
		cout << (ans * 4) % k << endl;
	}
	MPI_Finalize();
	return 0;
}
