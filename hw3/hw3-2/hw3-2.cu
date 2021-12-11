#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#define B 64
#define B_half 32

const int INF = ((1 << 30) - 1);
int *Dist = NULL;
int n, m, N;

inline void input(char* infile);
inline void output(char* outFileName);
inline int ceil(int a, int b);
inline void block_FW();
inline void floyed_warshall();
__global__ void phase_one(int *dst, int Round, int N);
__global__ void phase_two(int *dst, int Round, int N);
__global__ void phase_three(int *dst, int Round, int N);

__device__ int Min(int a, int b) {
	return min(a, b);
} 

int main(int argc, char* argv[]) {
	input(argv[1]);
	if (n <= 500) floyed_warshall();
	else block_FW();
	output(argv[2]);
	return 0;
}

inline void floyed_warshall() {
	for (int k = 0; k < n; k++) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (Dist[i*N+j] > Dist[i*N+k] + Dist[k*N+j]) {
					Dist[i*N+j] = Dist[i*N+k] + Dist[k*N+j];
				}
			}
		}
	}
}

inline int ceil(int a, int b) { return (a + b - 1) / b; }

inline void block_FW() {
	int round = ceil(n, B);
	int *dst = NULL;
	unsigned int size = N*N*sizeof(int);
	cudaHostRegister(Dist, size, cudaHostRegisterDefault);
	cudaMalloc(&dst, size);
	cudaMemcpy(dst, Dist, size, cudaMemcpyHostToDevice);
	
	int blocks = (N + B - 1) / B;
	dim3 block_dim(32, 32);
	dim3 grid_dim(blocks, blocks);
	for (int r = 0; r < round; ++r) {
		// phase 1
		phase_one<<<1, block_dim>>>(dst, r, N);
		// phase 2
		phase_two<<<blocks, block_dim>>>(dst, r, N);
		// phase 3
		phase_three<<<grid_dim, block_dim>>>(dst, r, N);
	}
	cudaMemcpy(Dist, dst, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dst);
}

__global__ void phase_one(int *dst, int Round, int N) {
	int i = threadIdx.y;
	int j = threadIdx.x;

	int place = Round * B * (N + 1) + i * N + j;
	int place_right = Round * B * (N + 1) + i * N + j + B_half;
	int place_down = Round * B * (N + 1) + (i + B_half) * N + j;
	int place_down_right = Round * B * (N + 1) + (i + B_half) * N + j + B_half;

	int i_B = i + B_half;
	int j_B = j + B_half;

	__shared__ int s[B][B];

	// load gloabal data to local memory
	s[i][j] = dst[place];
	s[i][j_B] = dst[place_right];
	s[i_B][j] = dst[place_down];
	s[i_B][j_B] = dst[place_down_right];
	__syncthreads();

	for (int k = 0; k < B; ++k) {
		s[i][j] = Min(s[i][k] + s[k][j], s[i][j]);
		s[i][j_B] = Min(s[i][k] + s[k][j_B], s[i][j_B]);
		s[i_B][j] = Min(s[i_B][k] + s[k][j], s[i_B][j]);
		s[i_B][j_B] = Min(s[i_B][k] + s[k][j_B], s[i_B][j_B]);
		__syncthreads();
	}
	dst[place] = s[i][j];
	dst[place_right] = s[i][j_B];
	dst[place_down] = s[i_B][j];
	dst[place_down_right] = s[i_B][j_B];
}

__global__ void phase_two(int *dst, int Round, int N) {
	if (blockIdx.x == Round) return;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int diagonal_place = Round * B * (N + 1) + i * N + j;
	int diagonal_place_right = Round * B * (N + 1) + i * N + j + B_half;
	int diagonal_place_down = Round * B * (N + 1) + (i + B_half) * N + j;
	int diagonal_place_down_right = Round * B * (N + 1) + (i + B_half) * N + j + B_half;

	int ver_place = blockIdx.x * B * N + Round * B + i * N + j;
	int ver_place_right = blockIdx.x * B * N + Round * B + i * N + j + B_half;
	int ver_place_down = blockIdx.x * B * N + Round * B + (i + B_half) * N + j;
	int ver_place_down_right = blockIdx.x * B * N + Round * B + (i + B_half) * N + j + B_half;

	int hor_place = Round * B * N + blockIdx.x * B + i * N + j;
	int hor_place_right = Round * B * N + blockIdx.x * B + i * N + j + B_half;
	int hor_place_down = Round * B * N + blockIdx.x * B + (i + B_half) * N + j;
	int hor_place_down_right = Round * B * N + blockIdx.x * B + (i + B_half) * N + j + B_half;

	int i_B = i + B_half;
	int j_B = j + B_half;

	__shared__ int s[B][B];
	__shared__ int ver[B][B];
	__shared__ int hor[B][B];

	s[i][j] = dst[diagonal_place];
	s[i][j_B] = dst[diagonal_place_right];
	s[i_B][j] = dst[diagonal_place_down];
	s[i_B][j_B] = dst[diagonal_place_down_right];

	ver[i][j] = dst[ver_place];
	ver[i][j_B] = dst[ver_place_right];
	ver[i_B][j] = dst[ver_place_down];
	ver[i_B][j_B] = dst[ver_place_down_right];

	hor[i][j] = dst[hor_place];
	hor[i][j_B] = dst[hor_place_right];
	hor[i_B][j] = dst[hor_place_down];
	hor[i_B][j_B] = dst[hor_place_down_right];

	__syncthreads();
	
	for (int k = 0; k < B; ++k) {
		ver[i][j] = Min(ver[i][j], ver[i][k] + s[k][j]);
		ver[i][j_B] = Min(ver[i][j_B], ver[i][k] + s[k][j_B]);
		ver[i_B][j] = Min(ver[i_B][j], ver[i_B][k] + s[k][j]);
		ver[i_B][j_B] = Min(ver[i_B][j_B], ver[i_B][k] + s[k][j_B]);

		hor[i][j] = Min(hor[i][j], s[i][k] + hor[k][j]);
		hor[i][j_B] = Min(hor[i][j_B], s[i][k] + hor[k][j_B]);
		hor[i_B][j] = Min(hor[i_B][j], s[i_B][k] + hor[k][j]);
		hor[i_B][j_B] = Min(hor[i_B][j_B], s[i_B][k] + hor[k][j_B]);
		
		__syncthreads();
	}

	dst[ver_place] = ver[i][j];
	dst[ver_place_right] = ver[i][j_B];
	dst[ver_place_down] = ver[i_B][j];
	dst[ver_place_down_right] = ver[i_B][j_B];

	dst[hor_place] = hor[i][j];
	dst[hor_place_right] = hor[i][j_B];
	dst[hor_place_down] = hor[i_B][j];
	dst[hor_place_down_right] = hor[i_B][j_B];
}

__global__ void phase_three(int *dst, int Round, int N) {
	if (blockIdx.x == Round || blockIdx.y == Round) return;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int self_place = blockIdx.y * B * N + blockIdx.x * B + i * N + j;
	int self_place_right = blockIdx.y * B * N + blockIdx.x * B + i * N + j + B_half;
	int self_place_down = blockIdx.y * B * N + blockIdx.x * B + (i + B_half) * N + j;
	int self_place_down_right = blockIdx.y * B * N + blockIdx.x * B + (i + B_half) * N + j + B_half;

	int a_place = blockIdx.y * B * N + Round * B + i * N + j;
	int a_place_right = blockIdx.y * B * N + Round * B + i * N + j + B_half;
	int a_place_down = blockIdx.y * B * N + Round * B + (i + B_half) * N + j;
	int a_place_down_right = blockIdx.y * B * N + Round * B + (i + B_half) * N + j + B_half;

	int b_place = Round * B * N + blockIdx.x * B + i * N + j;
	int b_place_right = Round * B * N + blockIdx.x * B + i * N + j + B_half;
	int b_place_down = Round * B * N + blockIdx.x * B + (i + B_half) * N + j;
	int b_place_down_right = Round * B * N + blockIdx.x * B + (i + B_half) * N + j + B_half;

	int i_B = i + B_half;
	int j_B = j + B_half;

	__shared__ int self[B][B];
	__shared__ int a[B][B];
	__shared__ int b[B][B];

	self[i][j] = dst[self_place];
	self[i][j_B] = dst[self_place_right];
	self[i_B][j] = dst[self_place_down];
	self[i_B][j_B] = dst[self_place_down_right];

	a[i][j] = dst[a_place];
	a[i][j_B] = dst[a_place_right];
	a[i_B][j] = dst[a_place_down];
	a[i_B][j_B] = dst[a_place_down_right];

	b[i][j] = dst[b_place];
	b[i][j_B] = dst[b_place_right];
	b[i_B][j] = dst[b_place_down];
	b[i_B][j_B] = dst[b_place_down_right];

	__syncthreads();

	#pragma unroll 32
	for (int k = 0; k < B; ++k) {
		self[i][j] = Min(a[i][k] + b[k][j], self[i][j]);
		self[i][j_B] = Min(a[i][k] + b[k][j_B], self[i][j_B]);
		self[i_B][j] = Min(a[i_B][k] + b[k][j], self[i_B][j]);
		self[i_B][j_B] = Min(a[i_B][k] + b[k][j_B], self[i_B][j_B]);
	}
	dst[self_place] = self[i][j];
	dst[self_place_right] = self[i][j_B];
	dst[self_place_down] = self[i_B][j];
	dst[self_place_down_right] = self[i_B][j_B];
}

inline void output(char* outFileName) {
	FILE* outfile = fopen(outFileName, "w");

	#pragma unroll 32
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i*N+j] >= INF) Dist[i*N+j] = INF;
		}
		fwrite(&Dist[i*N], sizeof(int), n, outfile);
	}
	fclose(outfile);
}

inline void input(char* infile) {
	int file = open(infile, O_RDONLY);
	int *ft = (int*)mmap(NULL, 2*sizeof(int), PROT_READ, MAP_PRIVATE, file, 0);
  n = ft[0];
	m = ft[1];
	int *pair = (int*)(mmap(NULL, (3 * m + 2) * sizeof(int), PROT_READ, MAP_PRIVATE, file, 0));

	if (n % B) N = n + (B - n % B);
	else N = n;

	Dist = (int*)malloc(N*N*sizeof(int));
	

	for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
			Dist[i*N+j] = INF;
			if (i == j) Dist[i*N+j] = 0;
		}
  }

	#pragma unroll
	for (int i = 0; i < m; ++i) {
		Dist[pair[i*3+2]*N+pair[i*3+3]]= pair[i*3+4];
	}
	close(file);
	munmap(pair, (3 * m + 2) * sizeof(int));
}