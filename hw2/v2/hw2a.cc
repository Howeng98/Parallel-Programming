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

using namespace std;

// 參數
struct data {
  int iters;
  int now;
  int height;
  int width;
  int chunk_size;
  double left;
  double right;
  double upper;
  double lower;
};

// thread pool, 紀錄 thread 數量跟現在有哪些 task
struct threadpool_t {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
    void (*function)(void*);
    bool shutdown;
    int thread_count;
    int work_size;
    int work_now;
    int chunk_size;

    // data
    int iters;
    int height;
    int width;
    double left;
    double right;
    double upper;
    double lower;
};

void mandelbrot_set(data* arg);
inline void* threadpool_thread(void* threadpool);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

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
  pool.work_size = width * height;
  pool.work_now = 0;
  pool.thread_count = ncpus;
  pool.threads = (pthread_t *)malloc(sizeof(pthread_t) * pool.thread_count);
  pool.shutdown = false;
  pthread_mutex_init(&(pool.lock), NULL);
  pthread_cond_init(&(pool.notify), NULL);
  pool.iters = iters;
  pool.left = left;
  pool.right = right;
  pool.lower = lower;
  pool.upper = upper;
  pool.width = width;
  pool.height = height;
  pool.chunk_size = 100;
  

  for (int i = 0; i < ncpus; i++) {
    pthread_create(&threads[i], NULL, threadpool_thread, &pool);
  }

  for (int i = 0; i < ncpus; i++) {
    pthread_join(threads[i], NULL);
  }

  write_png(filename, iters, width, height, image);
  pthread_exit(NULL);
}

inline void* threadpool_thread(void* threadpool) {
  threadpool_t *pool = (threadpool_t*)threadpool;
  data *arg;
  arg = new data;
  arg->height = pool->height;
  arg->width = pool->width;
  arg->upper = pool->upper;
  arg->lower = pool->lower;
  arg->iters = pool->iters;
  arg->left = pool->left;
  arg->right = pool->right;

  while (!pool->shutdown) {
    pthread_mutex_lock(&(pool->lock));
    arg->now = pool->work_now;
    pool->work_now = (pool->work_now + pool->chunk_size > pool->work_size) ? pool->work_size : pool->work_now + pool->chunk_size;

    if (pool->work_now == pool->work_size) pool->shutdown = true;
    pthread_mutex_unlock(&(pool->lock));
    arg->chunk_size = pool->work_now - arg->now;
    mandelbrot_set(arg);
  }

  pthread_exit(NULL);
}

inline void mandelbrot_set(data* arg) {
  int repeats = 0;
  double x = 0;
  double y = 0;
  double length_squred = 0;
  double temp;
  double x0;
  double y0;

  for (int i = 0; i < arg->chunk_size; i++) {
    x0 = ((arg->now + i) % arg->width) * ((arg->right - arg->left) / arg->width) + arg->left;
    y0 = ((arg->now + i) / arg->width) * ((arg->upper - arg->lower) / arg->height) + arg->lower;
    x = 0;
    y = 0;
    length_squred = 0;
    while (repeats < arg->iters && length_squred < 4) {
      temp = x * x - y * y + x0;
      y = 2 * x * y + y0;
      x = temp;
      length_squred = x * x + y * y;
      ++repeats;
    }
    image[arg->now+i] = repeats;
    repeats = 0;
  }
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
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