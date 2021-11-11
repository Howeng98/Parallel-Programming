#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <png.h>
#include <iostream>
#include <mpi.h>
#include <pthread.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <emmintrin.h>

using namespace std;

int* image;
struct _data {
  pthread_mutex_t lock;
  int width;
  int height;
  int proc_now;
  int proc_num;
};

inline void write_png(const char* filename, int iters, int width, int height, const int* buffer);
inline void* receive_png(void* Data);

int main(int argc, char** argv) {
  // get threads
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  int ncpus = CPU_COUNT(&cpu_set);
  pthread_t threads[ncpus];

  // get argument
  MPI_Init(&argc, &argv);
  const char* filename = argv[1];
  int iters = strtol(argv[2], 0, 10);
  double left = strtod(argv[3], 0);
  double right = strtod(argv[4], 0);
  double lower = strtod(argv[5], 0);
  double upper = strtod(argv[6], 0);
  int width = strtol(argv[7], 0, 10);
  int height = strtol(argv[8], 0, 10);

  // mpi argument
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;

  // assign task to every process
  image = (int*)malloc(height * width * sizeof(int));
  assert(image);
  memset(image, 0, width * height * sizeof(int));


  #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
    for (int i = rank; i < height; i+=size) {
      // mandelbrot set
      bool done[2];
      int repeats[2];
      int run_now[2];
      int record_now = 0;
      int start;
      double x;
      double y;
      double x0;
      double y0;
      double y0_add = (upper - lower) / (double)height;
      double x0_add = (right - left) / (double)width;
      double length_square;
      double constraint = 4;
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
      // see variable initialize
      x0 = left;
      y0 = lower + i * y0_add;
      x0_see[0] = x0;
      x0_see[1] = x0 + x0_add;
      y0_see[0] = y0_see[1] = y0;
      x_see[0] = x_see[1] = 0;
      y_see[0] = y_see[1] = 0;
      x_square_see[0] = x_square_see[1] = 0;
      y_square_see[0] = y_square_see[1] = 0;
      // other variable initialize
      done[0] = done[1] = false;
      start = i * width;
      run_now[0] = start;
      run_now[1] = start + 1;
      repeats[0] = repeats[1] = 0;
      record_now = 2;

      while (record_now <= width) {
        // see instuctions
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
          x0_see[0] = x0 + x0_add * record_now;
          done[0] = (record_now >= width) ? true : false;
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
          x0_see[1] = x0 + x0_add * record_now;
          done[1] = (record_now >= width) ? true : false;
          record_now += 1;
        }
      }

      // handle unfinished task
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

  if (rank == 0) {
    _data data;
    data.height = height;
    data.width = width;
    data.proc_now = 1;
    data.proc_num = size;
    pthread_mutex_init(&(data.lock), NULL);
    for (int i = 0; i < ncpus; i++) {
      pthread_create(&threads[i], NULL, receive_png, &data);
    }
    for (int i = 0; i < ncpus; i++) {
      pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&(data.lock));
  } else {
    MPI_Request req;
    MPI_Isend(image, width * height, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &status);
  }
  
  MPI_Finalize();
  return 0;
}

inline void* receive_png(void* Data) {
  _data *data = (_data*)Data;
  MPI_Status status;
  int total = data->height * data->width;
  int *tmp = (int*)malloc(total * sizeof(int));
  int proc_now = 1;
  int run;
  MPI_Request req;

  while (true) {
    pthread_mutex_lock(&(data->lock));
    proc_now = data->proc_now;
    data->proc_now += 1;
    pthread_mutex_unlock(&(data->lock));
    if (proc_now >= data->proc_num) break;

    MPI_Irecv(tmp, total, MPI_INT, proc_now, 0, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, &status);
    for (int i = proc_now; i < data->height; i += data->proc_num) {
      run = i * data->width;
      for (int j = 0; j < data->width; j++) {
        image[run + j] = tmp[run + j];
      }
    }
  }
  free(tmp);
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