#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <png.h>
#include <iostream>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <emmintrin.h>

using namespace std;

// 參數
struct data {
  void (*function)(void*);
  int iters;
  int height;
  int width;
  double left;
  double right;
  double upper;
  double lower;
  double x0_add;
  double y0_add;
  double x0;
  double y0;

  // changed data
  int row_now;
  int start;
};

// thread pool, 紀錄 thread 數量跟現在有哪些 task
struct threadpool_t {
  pthread_mutex_t lock;
  void (*function)(void*);
  bool shutdown;
  int thread_count;
  int row_now;

  // data
  int iters;
  int height;
  int width;
  double left;
  double right;
  double upper;
  double lower;
  double x0_add;
  double y0_add;
  double x0;
  double y0;
};

inline void mandelbrot_set(void* arg);
inline void* threadpool_thread(void* threadpool);
inline void write_png(const char* filename, int iters, int width, int height, const int* buffer);

// img
int* image;

int main(int argc, char** argv) {
  // thread 數量
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
  pthread_t threads[ncpus];

  // get argument
  const char* filename = argv[1];
  int iters = strtol(argv[2], 0, 10);
  double left = strtod(argv[3], 0);
  double right = strtod(argv[4], 0);
  double lower = strtod(argv[5], 0);
  double upper = strtod(argv[6], 0);
  int width = strtol(argv[7], 0, 10);
  int height = strtol(argv[8], 0, 10);
  image = (int*)malloc(width * height * sizeof(int));
  assert(image);

  threadpool_t pool;
  pool.thread_count = ncpus;
  pool.shutdown = false;
  pthread_mutex_init(&(pool.lock), NULL);
  pool.row_now = 0;
  pool.iters = iters;
  pool.left = left;
  pool.right = right;
  pool.lower = lower;
  pool.upper = upper;
  pool.width = width;
  pool.height = height;
  pool.x0_add = (right - left) / (double)width;
  pool.y0_add = (upper - lower) / (double)height;
  pool.x0 = left;
  pool.y0 = lower;

  for (int i = 0; i < ncpus; i++) {
    pthread_create(&threads[i], NULL, threadpool_thread, &pool);
  }

  for (int i = 0; i < ncpus; i++) {
    pthread_join(threads[i], NULL);
  }

  write_png(filename, iters, width, height, image);
  free(image);
  pthread_mutex_destroy(&pool.lock);
  pthread_exit(NULL);
}

inline void* threadpool_thread(void* threadpool) {
  threadpool_t *pool = (threadpool_t*)threadpool;
  data *arg;

  arg = new data;
  arg->function = mandelbrot_set;
  arg->height = pool->height;
  arg->width = pool->width;
  arg->upper = pool->upper;
  arg->lower = pool->lower;
  arg->iters = pool->iters;
  arg->left = pool->left;
  arg->right = pool->right;
  arg->x0_add = pool->x0_add;
  arg->y0_add = pool->y0_add;
  arg->x0 = pool->x0;

  while (true) {
    pthread_mutex_lock(&(pool->lock));
    arg->row_now = pool->row_now;
    pool->row_now += 1;
    pthread_mutex_unlock(&(pool->lock));
    arg->y0 = arg->lower + arg->row_now * arg->y0_add;
    arg->start = arg->row_now * arg->width;
    if (arg->row_now < arg->height) arg->function(arg);
    else break;
  }

  pthread_exit(NULL);
}

inline void mandelbrot_set(void* Arg) {
  data* arg = (data*)Arg;
  bool done[2];
  int repeats[2];
  int run_now[2];
  int record_now;
  double x0;
  double y0;
  double x0_add;
  double tmp;
  double constraint = 4;
  double two = 2;

  __m128d two_see;
  __m128d length_square_see;
  __m128d x_see;
  __m128d y_see;
  __m128d x0_see;
  __m128d y0_see;
  __m128d temp;

  x0_add = arg->x0_add;
  x0 = arg->x0;
  y0 = arg->y0;
  done[0] = done[1]  = false;
  run_now[0] = arg->start;
  run_now[1] = arg->start + 1;
  repeats[0] = repeats[1] = 0;
  record_now = 2;
  
  // sse vector
  two_see[0] = two_see[1] = two;
  x0_see[0] = x0;
  x0_see[1] = x0 + x0_add;
  x_see[0] = x_see[1] = 0;
  y0_see[0] = y0_see[1] = y0;
  y_see[0] = y_see[1] = 0;

  while (record_now <= arg->width) {
    // see instructions
    temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x_see, x_see), _mm_mul_pd(y_see, y_see)), x0_see);
    y_see = _mm_add_pd(_mm_mul_pd(two_see, _mm_mul_pd(x_see, y_see)), y0_see);
    x_see = temp;
    length_square_see = _mm_add_pd(_mm_mul_pd(x_see, x_see), _mm_mul_pd(y_see, y_see));
    ++repeats[0];
    ++repeats[1];

    if (length_square_see[0] >= constraint || repeats[0] >= arg->iters) {
      image[run_now[0]] = repeats[0];
      repeats[0] = 0;
      run_now[0] = arg->start + record_now;
      x_see[0] = 0;
      y_see[0] = 0;
      length_square_see[0] = 0;
      tmp = x0 + x0_add * record_now;
      x0_see[0] = tmp;
      done[0] = (record_now >= arg->width) ? true : false;
      record_now += 1;
    } if (length_square_see[1] >= constraint || repeats[1] >= arg->iters) {
      image[run_now[1]] = repeats[1];
      repeats[1] = 0;
      run_now[1] = arg->start + record_now;
      x_see[1] = 0;
      y_see[1] = 0;
      length_square_see[1] = 0;
      tmp = x0 + x0_add * record_now;
      x0_see[1] = tmp;
      done[1] = (record_now >= arg->width) ? true : false;
      record_now += 1;
    }
  }

  if (!done[0]) {
    double x = x_see[0];
    double y = y_see[0];
    double length_square = length_square_see[0];
    x0 = x0_see[0];

    while (repeats[0] < arg->iters && length_square < constraint) {
      tmp = x * x - y * y + x0;
      y = 2 * x * y + y0;
      x = tmp;
      length_square = x * x + y * y;
      ++repeats[0];
    }
    image[run_now[0]] = repeats[0];
  } if (!done[1]) {
    double x = x_see[1];
    double y = y_see[1];
    double length_square = length_square_see[1];
    x0 = x0_see[1];

    while (repeats[1] < arg->iters && length_square < constraint) {
      tmp = x * x - y * y + x0;
      y = 2 * x * y + y0;
      x = tmp;
      length_square = x * x + y * y;
      ++repeats[1];
    }
    image[run_now[1]] = repeats[1];
  }
}

inline void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
  FILE* fp = fopen(filename, "wb");
  assert(fp);
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  assert(png_ptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr);
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
          PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
  png_write_info(png_ptr, info_ptr);
  png_set_compression_level(png_ptr, 1);
  size_t row_size = 3 * width * sizeof(png_byte);
  png_bytep row = (png_bytep)malloc(row_size);
  for (int y = 0; y < height; ++y) {
    memset(row, 0, row_size);
    for (int x = 0; x < width; ++x) {
      int p = buffer[(height - 1 - y) * width + x];
      png_bytep color = row + x * 3;
      if (p != iters) {
        if (p & 16) {
            color[0] = 240;
            color[1] = color[2] = p % 16 * 16;
        } else {
            color[0] = p % 16 * 16;
        }
      }
    }
    png_write_row(png_ptr, row);
  }
  free(row);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
}