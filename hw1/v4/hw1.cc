#include <iostream>
#include <boost/sort/spreadsort/float_sort.hpp>
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

// for right
void exchange_right(int neighbor_total, int total) {
  // early terminate
  if (temp_buffer[neighbor_total-1] <= data[0]) return;

  for (int i = total-1, run_n = neighbor_total-1, run = total-1; i >= 0; i--) {
    put_buffer[i] = (run_n >= 0 && data[run] < temp_buffer[run_n]) ? temp_buffer[run_n--] : data[run--];
  }

  swap(data, put_buffer);
}

// for left
void exchange_left(int neighbor_total, int total) {
  if (temp_buffer[0] >= data[total-1]) return;

  for (int i = 0, run_n = 0, run = 0; i < total; i++) {
    put_buffer[i] = (run_n < neighbor_total && data[run] > temp_buffer[run_n]) ? temp_buffer[run_n++] : data[run++];
  }

  swap(data, put_buffer);
}

int main(int argc, char** argv) {
  int rank, size, n, flag;
  int start;
  int total;
  int left_total, right_total;
  int q, r; // quotient and remainder
  unsigned long long offset;
  bool result, now;

  left_total = right_total = 0;
  result = now = false;
  n = atoll(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;
  MPI_File f;
  MPI_Group old_group, new_group;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  // start time
  //double start_time = MPI_Wtime();

  // assign number to every process
  // if the process num larger than n, remove necessary process
  if (size > n) {
    // get the whole process member
    MPI_Comm_group(MPI_COMM_WORLD, &old_group);

    // remove necessary process
    int range[][3] = {{n, size-1, 1}};
    MPI_Group_range_excl(old_group, 1, range, &new_group);

    // create new comm
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &mpi_comm);

    // terminate uncessary process
    if(mpi_comm == MPI_COMM_NULL){
      MPI_Finalize();
      exit(0);
    }
    size = n;
    total = 1;
    start = rank;
    left_total = right_total = 1;
  } else {
    q = n / size;
    r = n % size;
    bool judge = (rank < r) ? true : false;
    total = q;
    start = q * rank;
    if (judge) {
      // if judge is true, total number plus 1
      total += 1;
      start += rank;
      // determine left total number and right total number
      left_total = q + 1;
      if (rank != size - 1 && rank + 1 < r) right_total = q + 1;
      else if (rank != size -1) right_total = q;
    } else {
      // determine left total number and right total number
      right_total = q;
      start += r;
      if (rank != 0 && rank - 1 < r) left_total = q + 1;
      else if (rank != 0) left_total = q;
    }
  }

  offset = start * sizeof(float);
  data = (float*)malloc(sizeof(float)*(n / size + size));
  temp_buffer = (float*)malloc(sizeof(float)*(n / size + size));
  put_buffer = (float*)malloc(sizeof(float)*(n / size + size));

  // read input
  int check = MPI_File_open(mpi_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
  if (check != MPI_SUCCESS) {
    cout << "open file error" << endl;
    MPI_Abort(mpi_comm, check);
  }
  MPI_File_read_at(f, offset, data, total, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&f);

  // float_sort
  float_sort(data, data+total, rightshift());

  // start odd-even sort
  flag = 0;
  for(int i = 0; i < size + 1; i++) {
    if (flag == 0) { // even sort
     if (rank % 2 == 1) {
        MPI_Sendrecv(data, total, MPI_FLOAT, rank - 1, 0, temp_buffer, left_total, MPI_FLOAT, rank - 1, 0, mpi_comm, &status);
        exchange_right(left_total, total);
      } else if (rank != size-1) {
        MPI_Sendrecv(data, total, MPI_FLOAT, rank + 1, 0, temp_buffer, right_total, MPI_FLOAT, rank + 1, 0, mpi_comm, &status);
        exchange_left(right_total, total);
      }
    } else { // odd sort
      if (rank % 2 == 0 && rank != 0) {
        MPI_Sendrecv(data, total, MPI_FLOAT, rank - 1, 0, temp_buffer, left_total, MPI_FLOAT, rank - 1, 0, mpi_comm, &status);
        exchange_right(left_total, total);
      } else if (rank % 2 == 1 && rank != size - 1){
        MPI_Sendrecv(data, total, MPI_FLOAT, rank + 1, 0, temp_buffer, right_total, MPI_FLOAT, rank + 1, 0, mpi_comm, &status);
        exchange_left(right_total, total);
      }
    }

    flag = (flag == 0) ? 1 : 0;
  }

  // write back 
  check = MPI_File_open(mpi_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
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