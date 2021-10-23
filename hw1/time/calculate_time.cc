#include "calculate_time.h"


CalculateTime::CalculateTime() {
  this->IO_time = 0;
  this->CPU_time = 0;
  this->COMM_time = 0;
  time_point_IO = 0;
  time_point_COMM = 0;
  time_point_CPU = 0;
}

CalculateTime::~CalculateTime() {
  // do nothing
}

void CalculateTime::record_cpu() {
  this->time_point_CPU = MPI_Wtime();
}

void CalculateTime::record_comm() {
  this->time_point_COMM = MPI_Wtime();
}

void CalculateTime::record_io() {
  this->time_point_IO = MPI_Wtime();
}

void CalculateTime::update_cpu() {
  this->CPU_time += (MPI_Wtime() - this->time_point_CPU);
}

void CalculateTime::update_comm() {
  this->COMM_time += (MPI_Wtime() - this->time_point_COMM);
}

void CalculateTime::update_io() {
  this->IO_time += (MPI_Wtime() - this->time_point_IO);
}

void CalculateTime::print_result() {
  cout << this->CPU_time << endl;
  cout << this->COMM_time << endl;
  cout << this->IO_time << endl;
}