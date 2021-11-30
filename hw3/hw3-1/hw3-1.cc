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
      #pragma unroll 50
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

void input(char* infile) {
  FILE* file = fopen(infile, "rb");
  fread(&n, sizeof(int), 1, file);
  fread(&m, sizeof(int), 1, file);

  for (int i = 0; i < n; ++i) {
    #pragma unroll 3
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        Dist[i][j] = 0;
      } else {
        Dist[i][j] = INF;
      }
    }
  }

  int pair[3];
  #pragma unroll 3
  for (int i = 0; i < m; ++i) {
    fread(pair, sizeof(int), 3, file);
    Dist[pair[0]][pair[1]] = pair[2];
  }
  fclose(file);
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