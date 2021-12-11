#include <iostream>
#include <omp.h>

using namespace std;

inline void input(char* inFileName);
inline void output(char* outFileName);

const int INF = ((1 << 30) - 1);
const int V = 50010;
static int Dist[V][V];
int n, m;

int main(int argc, char *argv[]) {
  input(argv[1]);
  for (int k = 0; k < n; k++) {
  #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      #pragma unroll 4
      for (int j = 0; j < n; j++) {
        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
          Dist[i][j] = Dist[i][k] + Dist[k][j];
        }
      }
    }
  }
  output(argv[2]);
  return 0;
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

void input(char* infile) {
  int file = open(infile, O_RDONLY);
	int *ft = (int*)mmap(NULL, 2*sizeof(int), PROT_READ, MAP_PRIVATE, file, 0);
  n = ft[0];
	m = ft[1];

  int *pair = (int*)(mmap(NULL, (3 * m + 2) * sizeof(int), PROT_READ, MAP_PRIVATE, file, 0));

  Dist = (int*)malloc(n*n*sizeof(int));

	for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
			Dist[i][j] = INF;
			if (i == j) Dist[i][j] = 0;
		}
  }

  #pragma unroll
	for (int i = 0; i < m; ++i) {
		Dist[pair[i*3+2]][pair[i*3+3]]= pair[i*3+4];
	}
	close(file);
	munmap(pair, (3 * m + 2) * sizeof(int));
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        #pragma unroll 3
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}