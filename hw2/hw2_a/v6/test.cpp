#include <iostream>
#include <pthread.h>
#include <emmintrin.h>
#include <time.h>  

using namespace std;

int main(int argc, char** argv) {
  __m128d x, y, z;
  clock_t start, End;
  double a = 100;
  double b = 200;
  y[0] = 100;
  y[1] = 200;
  x[0] = 0;
  x[1] = 0;
  z[0] = 500;
  z[1] = 600;

  start = clock();
  for (int i = 0; i < 100000000; i++) {
    x = y + z;
  }
  End = clock();
  
  cout << "assign cost time = " << End - start << endl;
  cout << x[0] << ' ' << x[1] << endl;

  x[0] = 0;
  x[1] = 0;
  y[0] = 100;
  y[1] = 200;
  z[0] = 500;
  z[1] = 600;
 
  start = clock();
  for (int i = 0; i < 100000000; i++) {
    x = _mm_add_pd(y, z);
  }
  End = clock();

  cout << "mm_set cost time = " << End - start << endl;
  cout << x[0] << ' ' << x[1] << endl;

  return 0;
}