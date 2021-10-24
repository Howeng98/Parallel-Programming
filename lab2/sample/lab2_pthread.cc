#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <pthread.h>

using namespace std;

unsigned long long part = 0;

typedef struct _data {
	unsigned long long r;
	unsigned long long k;
	unsigned long long squre_long;
	unsigned long long size;
	unsigned long long result;
	pthread_mutex_t lock;
} data;

void* calculate_pixel(void* Input) {
	data* input = (data*)Input;

	unsigned long long rank;
	unsigned long long pixels = 0;
	unsigned long long y;
	unsigned long long begin;
	unsigned long long end;
	unsigned long long a;
	unsigned long long b;
	unsigned long long add;
	unsigned long long sq = input->r * input->r;

	rank = part++;
	begin = (input->squre_long / input->size) * rank;
	end = begin + input->squre_long / input->size;
	a = ceil(sqrtl(sq-begin*begin));
	b = (a-1)*(a-1);
	add = 2 * (a - input->squre_long);
	if (rank == input->size-1) end = input->squre_long;

	for (unsigned long long x = begin; x < end; x++) {
		if (sq - x * x <= b) {
			a -= 1;
			b = (a-1) * (a-1);
			add = 2 * (a - input->squre_long); 
		}
		pixels += add;
	}
	pixels %= input->k;

	pthread_mutex_lock(&(input->lock));
	input->result += pixels;
	input->result %= input->k;
	pthread_mutex_unlock(&(input->lock));

	pthread_exit(NULL);
	return 0;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long sq = r*r;
	unsigned long long squre_long = ceil(r / sqrtl(2));
	unsigned long long squre_size = (squre_long * squre_long) % k;
	unsigned long long ans = 0;
	// thread 數量
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	pthread_t threads[ncpus];
	data Input;
	Input.r = r;
	Input.k = k;
	Input.squre_long = squre_long;
	Input.result = 0;
	Input.size = ncpus;
	pthread_mutex_init(&(Input.lock), NULL);

	for (int i = 0; i < ncpus; i++) {
		pthread_create(&threads[i], NULL, calculate_pixel, &Input);
	}
	for (int i = 0; i < ncpus; i++) {
		pthread_join(threads[i], NULL);
	}

	ans = (Input.result + squre_size) % k;
	ans *= 4;
	ans %= k;
	printf("%llu\n", ans);
	pthread_exit(NULL);
}
