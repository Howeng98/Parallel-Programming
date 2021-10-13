#include <iostream>
#include <math.h>
#include <mpi.h>
#include <algorithm>

using namespace std;

float* data;
float* temp_buffer;
float* put_buffer;

void exchange(int flag, int neighbor_total, int n, int rank) {
  int self, neighbor, run, neighbor_run;
  bool tmp = false;
  float num;
  MPI_Status status;
  self = neighbor = run = neighbor_run = 0;
  
  // check if no need to change
  if (data[0] >= temp_buffer[neighbor_total-1]) {
    tmp = true;
    MPI_Send(&tmp, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD);
    return;
  } else {
    tmp = false;
    MPI_Send(&tmp, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD);
  }

  // start to sort element
  while (self < n || neighbor < neighbor_total) {
    if (neighbor_run < neighbor_total) {
      if (data[self] <= temp_buffer[neighbor]) {
        num = data[self];
        self += 1;
      } else {
        num = temp_buffer[neighbor];
        neighbor += 1;
      }
      put_buffer[neighbor_run] = num;
      neighbor_run += 1;
    } else {
      if (self == n) {
        num = temp_buffer[neighbor];
        neighbor += 1;
      } else if (neighbor == neighbor_total) {
        num = data[self];
        self += 1;
      } else if (data[self] <= temp_buffer[neighbor]) {
        num = data[self];
        self += 1;
      } else {
        num = temp_buffer[neighbor];
        neighbor += 1;
      }
      data[run] = num;
      run += 1;
    }
  }
  MPI_Send(put_buffer, neighbor_total, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  int rank, size, n, flag;
  bool odd = false;
  bool even = false;
  bool result = false;
  double starttime, endtime;
  n = atoll(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Status status;
  MPI_File f;
  int start;
  int end;
  int total;
  unsigned long long offset;
  int neighbor_total;

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

  // sort 
  sort(data, data+total);

  // start odd-even sort
  flag = 0;
  for (int j = 0; j < size+1; j++) {
    if (flag == 0) { // even sort
      if (rank % 2 == 0 && rank == size -1) {
        odd = true;
      } else if (rank % 2 == 1) {
        MPI_Recv(&neighbor_total, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(temp_buffer, neighbor_total, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
        exchange(flag, neighbor_total, total, rank);
      } else {
        MPI_Send(&total, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(data, total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        bool tmp;
        MPI_Recv(&tmp, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &status);
        if (!tmp) {
          MPI_Recv(data, total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
      }
    } else { // odd sort
      if (rank == 0 || (rank == size - 1 && rank % 2 == 1)) {
        even = true;
      } else if (rank % 2 == 0 && rank != 0) {
        MPI_Recv(&neighbor_total, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(temp_buffer, neighbor_total, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
        exchange(flag, neighbor_total, total, rank);
      } else {
        MPI_Send(&total, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(data, total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        bool tmp;
        MPI_Recv(&tmp, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &status);
        if (!tmp) {
          MPI_Recv(data, total, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
      }
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
  if (rank == 0) {
    cout << "total time: " << MPI_Wtime() - start_time << endl;
  }
  MPI_Finalize();

  return 0;
}