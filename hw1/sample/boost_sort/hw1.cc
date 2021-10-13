#include <iostream>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <vector>
#include <deque>
#include <mpi.h>
#define CAST_TYPE int
#define DATA_TYPE float

using namespace std;
using namespace boost::sort::spreadsort;

float* data;
float* temp_buffer;
float* put_buffer;

// make float sort in acending order
struct rightshift{
inline CAST_TYPE operator()(const DATA_TYPE &x, const unsigned offset) const {
    return float_mem_cast<DATA_TYPE, CAST_TYPE>(x) >> offset;
  }
};

// odd-even sort
bool exchange(int flag, int neighbor_total, int total, int rank) {
  if ((flag == 0 && rank % 2 == 1) || (flag == 1 && rank % 2 == 0)) {
    if (flag == 0 && rank % 2 == 1 && temp_buffer[neighbor_total-1] <= data[0]) return true;
    else if (flag == 1 && rank % 2 == 0 && temp_buffer[neighbor_total-1] <= data[0]) return true;
    int run_n, run;
    run_n = neighbor_total - 1;
    run = total - 1;
    for (int i = total - 1; i >= 0; i--) {
      if (run_n >= 0 && temp_buffer[run_n] > data[run]) {
        put_buffer[i] = temp_buffer[run_n];
        run_n -= 1;
      } else {
        put_buffer[i] = data[run];
        run -= 1;
      }
    }
  } else {
    if (flag == 0 && rank % 2 == 0 && temp_buffer[0] >= data[total-1]) return true;
    else if (flag == 1 && rank % 2 == 1 && temp_buffer[0] >= data[total-1]) return true;
    int run_n, run;
    run_n = 0;
    run = 0;
    for (int i = 0; i < total; i++) {
      if (run_n <= neighbor_total && temp_buffer[run_n] < data[run]) {
        put_buffer[i] = temp_buffer[run_n];
        run_n += 1;
      } else {
        put_buffer[i] = data[run];
        run += 1;
      }
    }
  }

  for (int i = 0; i < total; i++) data[i] = put_buffer[i];

  return false;
}

int main(int argc, char** argv) {
  int rank, size, n, flag;
  bool odd = false;
  bool even = false;
  bool result = false;
  bool now = false;
  bool left_check, right_check;
  double starttime, endtime;
  int start;
  int end;
  int total;
  unsigned long long offset;
  int neighbor_total, left_total, right_total;

  left_total = right_total = 0;
  left_check = right_check = false;
  n = atoll(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;
  MPI_File f;
  bool all_status[size] = {false};

  // start time
  double start_time = MPI_Wtime();

  start = (n / size) * rank;
  end = start + n / size;
  if (rank == size-1) end = n;
  total = end - start;
  offset = start * sizeof(float);

  data = new float[n / size + size];
  temp_buffer = new float[n / size + size];
  put_buffer = new float[n/ size + size];

  // read input
  int check = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
  if (check != MPI_SUCCESS) {
    cout << "open file error" << endl;
    MPI_Abort(MPI_COMM_WORLD, check);
  }
  MPI_File_read_at(f, offset, data, total, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&f);

  // float_sort
  float_sort(data, data+total, rightshift());

  // start odd-even sort
  flag = 0;
  while(!result) {
    if (flag == 0) { // even sort
      if (rank % 2 == 0 && rank == size -1) {
        odd = true;
      } else if (rank % 2 == 1) {
        if (!left_check) {
          left_check = true;
          MPI_Sendrecv(&total, 1, MPI_INT, rank - 1, 0, &left_total, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if (left_total != 0 && total != 0) {
          MPI_Sendrecv(data, total, MPI_FLOAT, rank - 1, 0, temp_buffer, left_total, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
          odd = exchange(flag, left_total, total, rank);
        }
      } else {
        if (!right_check) {
          right_check = true;
          MPI_Sendrecv(&total, 1, MPI_INT, rank + 1, 0, &right_total, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
        if (right_total != 0 && total != 0) {
          MPI_Sendrecv(data, total, MPI_FLOAT, rank + 1, 0, temp_buffer, right_total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
          odd = exchange(flag, right_total, total, rank);
        }
      }
    } else { // odd sort
      if (rank == 0 || (rank == size - 1 && rank % 2 == 1)) {
        even = true;
      } else if (rank % 2 == 0) {
        if (!left_check) {
          left_check = true;
          MPI_Sendrecv(&total, 1, MPI_INT, rank - 1, 0, &left_total, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if (left_total != 0 && total != 0) {
          MPI_Sendrecv(data, total, MPI_FLOAT, rank - 1, 0, temp_buffer, left_total, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
          even = exchange(flag, left_total, total, rank);
        }
      } else {
        if (!right_check) {
          right_check = true;
          MPI_Sendrecv(&total, 1, MPI_INT, rank + 1, 0, &right_total, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
        if (right_total != 0 && total != 0) {
          MPI_Sendrecv(data, total, MPI_FLOAT, rank + 1, 0, temp_buffer, right_total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
          even = exchange(flag, right_total, total, rank);
        }
      }
    }
    if (flag == 1) {
      now = even & odd;
      MPI_Allreduce(&now, &result, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    flag = (flag == 0) ? 1 : 0;
  }

  // write back 
  check = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
  if (check != MPI_SUCCESS) {
    cout << "open file error" << endl;
    MPI_Abort(MPI_COMM_WORLD, check);
  }
  MPI_File_write_at(f, offset, data, total, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&f);

  // calculate total time
  // if (rank == 0) {
  //   cout << "total time: " << MPI_Wtime() - start_time << endl;
  // }

  MPI_Finalize();

  return 0;
}