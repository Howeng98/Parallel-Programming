#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <png.h>
#include <iostream>
#include <pthread.h>
#include <string.h>
#include <emmintrin.h>
#include <math.h>
#include <mpi.h>

using namespace std;

// thread pool, 紀錄 thread 數量跟現在有哪些 task
struct threadpool_t {
  pthread_mutex_t lock;
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

  // for cal
  double lowest_time;
  double highest_time;
  double average_time;
  double total_time;
};

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

  MPI_Init(&argc, &argv);

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

  pool.total_time = 0;
  pool.lowest_time = 0;
  pool.highest_time = 0;

  for (int i = 0; i < ncpus; i++) {
    pthread_create(&threads[i], NULL, threadpool_thread, &pool);
  }

  for (int i = 0; i < ncpus; i++) {
    pthread_join(threads[i], NULL);
  }
  pool.average_time = pool.total_time / ncpus;
  cout << "average time: " << pool.average_time << endl;
  cout << "highest time: " << pool.highest_time << endl;
  cout << "lowest time: " << pool.lowest_time << endl;

  write_png(filename, iters, width, height, image);

  free(image);
  pthread_mutex_destroy(&pool.lock);
  pthread_exit(NULL);
}

inline void* threadpool_thread(void* threadpool) {
  threadpool_t *pool = (threadpool_t*)threadpool;
  double start_time = MPI_Wtime();
  double thread_time;

  bool done[2];
  int repeats[2];
  int run_now[2];
  int start;
  int record_now = 0;
  int row_now;
  int iters = pool->iters;
  double x;
  double y;
  double x0;
  double y0;
  double length_square;
  double constraint = 4;
  double two = 2;
  double x_square;
  double y_square;

  __m128d two_see;
  __m128d length_square_see;
  __m128d x_see;
  __m128d y_see;
  __m128d x0_see;
  __m128d y0_see;
  __m128d x_square_see;
  __m128d y_square_see;

  two_see[0] = two_see[1] = 2;
  
  while (true) {
    pthread_mutex_lock(&(pool->lock));
    row_now = pool->row_now;
    pool->row_now += 1;
    pthread_mutex_unlock(&(pool->lock));

    // terminate
    if (row_now >= pool->height) break;

    // update parameter
    y0 = pool->lower + row_now * pool->y0_add;
    x0 = pool->left;
    x0_see[0] = x0;
    x0_see[1] = x0 + pool->x0_add;
    x_see[0] = x_see[1] = 0;
    y0_see[0] = y0_see[1] = y0;
    y_see[0] = y_see[1] = 0;
    x_square_see[0] = x_square_see[1] = 0;
    y_square_see[0] = y_square_see[1] = 0;

    done[0] = done[1]  = false;
    start = row_now * pool->width;
    run_now[0] = start;
    run_now[1] = start + 1;
    repeats[0] = repeats[1] = 0;
    record_now = 2;

    while (record_now <= pool->width) {
      // see instructions
      y_see = _mm_add_pd(_mm_mul_pd(two_see, _mm_mul_pd(x_see, y_see)), y0_see);
      x_see = _mm_add_pd(_mm_sub_pd(x_square_see, y_square_see), x0_see);
      x_square_see = _mm_mul_pd(x_see, x_see);
      y_square_see = _mm_mul_pd(y_see, y_see);
      length_square_see = _mm_add_pd(x_square_see, y_square_see);
      ++repeats[0];
      ++repeats[1];

      if (length_square_see[0] >= constraint || repeats[0] >= iters) {
        image[run_now[0]] = repeats[0];
        repeats[0] = 0;
        run_now[0] = start + record_now;
        x_see[0] = 0;
        y_see[0] = 0;
        x_square_see[0] = 0;
        y_square_see[0] = 0;
        length_square_see[0] = 0;
        x0_see[0] = x0 + pool->x0_add * record_now;
        done[0] = (record_now >= pool->width) ? true : false;
        record_now += 1;
      } if (length_square_see[1] >= constraint || repeats[1] >= iters) {
        image[run_now[1]] = repeats[1];
        repeats[1] = 0;
        run_now[1] = start + record_now;
        x_see[1] = 0;
        y_see[1] = 0;
        x_square_see[1] = 0;
        y_square_see[1] = 0;
        length_square_see[1] = 0;
        x0_see[1] = x0 + pool->x0_add * record_now;
        done[1] = (record_now >= pool->width) ? true : false;
        record_now += 1;
      }
    }
    if (!done[0]) {
      x = x_see[0];
      y = y_see[0];
      x_square = x_square_see[0];
      y_square = y_square_see[0];
      length_square = length_square_see[0];
      x0 = x0_see[0];

      while (repeats[0] < iters && length_square < constraint) {
        y = 2 * x * y + y0;
        x = x_square - y_square + x0;
        x_square = x * x;
        y_square = y * y;
        length_square = x_square + y_square;
        ++repeats[0];
      }
      image[run_now[0]] = repeats[0];
    } if (!done[1]) {
      x = x_see[1];
      y = y_see[1];
      x_square = x_square_see[1];
      y_square = y_square_see[1];
      length_square = length_square_see[1];
      x0 = x0_see[1];

      while (repeats[1] < iters && length_square < constraint) {
        y = 2 * x * y + y0;
        x = x_square - y_square + x0;
        x_square = x * x;
        y_square = y * y;
        length_square = x_square + y_square;
        ++repeats[1];
      }
      image[run_now[1]] = repeats[1];
    }
  }

  thread_time = MPI_Wtime() - start_time;
  pthread_mutex_lock(&(pool->lock));
  pool->total_time += thread_time;
  pool->lowest_time = (pool->lowest_time == 0 || pool->lowest_time > thread_time) ? thread_time : pool->lowest_time;
  pool->highest_time = (pool->highest_time < thread_time) ? thread_time : pool->highest_time;
  pthread_mutex_unlock(&(pool->lock));
  pthread_exit(NULL);
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