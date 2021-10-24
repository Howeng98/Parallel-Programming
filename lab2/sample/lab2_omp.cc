#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int rank, size;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long result = 0;
	unsigned long long sq = r*r;
	unsigned long long squre_long = ceil(r / sqrtl(2));
	unsigned long long squre_size = (squre_long * squre_long) % k;
	

	#pragma omp parallel shared(result, sq, squre_long, squre_size, r, k) 
	{
		size = omp_get_num_threads();
		rank = omp_get_thread_num();
		unsigned long long begin, end, a, b, add, pixels;

		pixels = 0;
		begin = (squre_long / size) * rank;
		end = (rank == size-1) ? squre_long : begin + squre_long / size;
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

	result = (result + squre_size) % k;

	cout << (result * 4) % k << endl;
}
