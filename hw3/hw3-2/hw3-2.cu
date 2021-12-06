#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define B 64
#define B_half 32

const int INF = ((1 << 30) - 1);
int *Dist = NULL;
int n, m, N;

inline void input(char* infile);
inline void output(char* outFileName);
inline int ceil(int a, int b);
inline void block_FW();
__global__ void phase_one(int *dst, int Round, int N);
__global__ void phase_two(int *dst, int Round, int N);
__global__ void phase_three(int *dst, int Round, int N);

__device__ int Min(int a, int b) {
	return min(a, b);
} 

int main(int argc, char* argv[]) {
	input(argv[1]);
	block_FW();
	output(argv[2]);
	return 0;
}

inline int ceil(int a, int b) { return (a + b - 1) / b; }

inline void block_FW() {
	int round = ceil(n, B);
	int *dst = NULL;
	cudaMalloc(&dst, N*N*sizeof(int));
	cudaMemcpy(dst, Dist, N*N*sizeof(int), cudaMemcpyHostToDevice);
	
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

	__shared__ int s[B][B];

	// load gloabal data to local memory
	s[i][j] = dst[place];
	s[i][j+B_half] = dst[place_right];
	s[i+B_half][j] = dst[place_down];
	s[i+B_half][j+B_half] = dst[place_down_right];
	__syncthreads();

	for (int k = 0; k < B; ++k) {
		s[i][j] = Min(s[i][k] + s[k][j], s[i][j]);
		s[i][j+B_half] = Min(s[i][k] + s[k][j+B_half], s[i][j+B_half]);
		s[i+B_half][j] = Min(s[i+B_half][k] + s[k][j], s[i+B_half][j]);
		s[i+B_half][j+B_half] = Min(s[i+B_half][k] + s[k][j+B_half], s[i+B_half][j+B_half]);
		__syncthreads();
	}
	dst[place] = s[i][j];
	dst[place_right] = s[i][j+B_half];
	dst[place_down] = s[i+B_half][j];
	dst[place_down_right] = s[i+B_half][j+B_half];
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

	__shared__ int s[B][B];
	__shared__ int ver[B][B];
	__shared__ int hor[B][B];

	s[i][j] = dst[diagonal_place];
	s[i][j+B_half] = dst[diagonal_place_right];
	s[i+B_half][j] = dst[diagonal_place_down];
	s[i+B_half][j+B_half] = dst[diagonal_place_down_right];

	ver[i][j] = dst[ver_place];
	ver[i][j+B_half] = dst[ver_place_right];
	ver[i+B_half][j] = dst[ver_place_down];
	ver[i+B_half][j+B_half] = dst[ver_place_down_right];

	hor[i][j] = dst[hor_place];
	hor[i][j+B_half] = dst[hor_place_right];
	hor[i+B_half][j] = dst[hor_place_down];
	hor[i+B_half][j+B_half] = dst[hor_place_down_right];

	__syncthreads();
	
	#pragma unroll
	for (int k = 0; k < B; ++k) {
		ver[i][j] = Min(ver[i][j], ver[i][k] + s[k][j]);
		ver[i][j+B_half] = Min(ver[i][j+B_half], ver[i][k] + s[k][j+B_half]);
		ver[i+B_half][j] = Min(ver[i+B_half][j], ver[i+B_half][k] + s[k][j]);
		ver[i+B_half][j+B_half] = Min(ver[i+B_half][j+B_half], ver[i+B_half][k] + s[k][j+B_half]);

		hor[i][j] = Min(hor[i][j], s[i][k] + hor[k][j]);
		hor[i][j+B_half] = Min(hor[i][j+B_half], s[i][k] + hor[k][j+B_half]);
		hor[i+B_half][j] = Min(hor[i+B_half][j], s[i+B_half][k] + hor[k][j]);
		hor[i+B_half][j+B_half] = Min(hor[i+B_half][j+B_half], s[i+B_half][k] + hor[k][j+B_half]);
	}

	dst[ver_place] = ver[i][j];
	dst[ver_place_right] = ver[i][j+B_half];
	dst[ver_place_down] = ver[i+B_half][j];
	dst[ver_place_down_right] = ver[i+B_half][j+B_half];

	dst[hor_place] = hor[i][j];
	dst[hor_place_right] = hor[i][j+B_half];
	dst[hor_place_down] = hor[i+B_half][j];
	dst[hor_place_down_right] = hor[i+B_half][j+B_half];
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

	__shared__ int self[B][B];
	__shared__ int a[B][B];
	__shared__ int b[B][B];

	self[i][j] = dst[self_place];
	self[i][j+B_half] = dst[self_place_right];
	self[i+B_half][j] = dst[self_place_down];
	self[i+B_half][j+B_half] = dst[self_place_down_right];

	a[i][j] = dst[a_place];
	a[i][j+B_half] = dst[a_place_right];
	a[i+B_half][j] = dst[a_place_down];
	a[i+B_half][j+B_half] = dst[a_place_down_right];

	b[i][j] = dst[b_place];
	b[i][j+B_half] = dst[b_place_right];
	b[i+B_half][j] = dst[b_place_down];
	b[i+B_half][j+B_half] = dst[b_place_down_right];

	__syncthreads();

	#pragma unroll(32)
	for (int k = 0; k < B; ++k) {
		self[i][j] = Min(a[i][k] + b[k][j], self[i][j]);
		self[i][j+B_half] = Min(a[i][k] + b[k][j+B_half], self[i][j+B_half]);
		self[i+B_half][j] = Min(a[i+B_half][k] + b[k][j], self[i+B_half][j]);
		self[i+B_half][j+B_half] = Min(a[i+B_half][k] + b[k][j+B_half], self[i+B_half][j+B_half]);
	}
	dst[self_place] = self[i][j];
	dst[self_place_right] = self[i][j+B_half];
	dst[self_place_down] = self[i+B_half][j];
	dst[self_place_down_right] = self[i+B_half][j+B_half];
}

inline void output(char* outFileName) {
	FILE* outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i*N+j] >= INF) Dist[i*N+j] = INF;
		}
		fwrite(&Dist[i*N], sizeof(int), n, outfile);
	}
	fclose(outfile);
}

inline void input(char* infile) {
	FILE* file = fopen(infile, "rb");
	fread(&n, sizeof(int), 1, file);
	fread(&m, sizeof(int), 1, file);

	if (n % B) N = n + (B - n % B);
	else N = n;
	cudaHostAlloc(&Dist, N*N*sizeof(int), 1);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (i == j) {
				Dist[i*N+j] = 0;
			} else {
				Dist[i*N+j] = INF;
			}
		}
	}

	int pair[3];
	for (int i = 0; i < m; ++i) {
		fread(pair, sizeof(int), 3, file);
		Dist[pair[0]*N+pair[1]]= pair[2];
	}

	fclose(file);
}