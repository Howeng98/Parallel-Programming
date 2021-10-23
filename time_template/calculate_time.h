#include <iostream>
#include <fstream>
#include <mpi.h>

using namespace std;

class CalculateTime {
  public:
    // constructer
    CalculateTime();
    // destructer
    ~CalculateTime();
    // record time point
    void record_cpu();
    void record_comm();
    void record_io();
    // update time
    void update_cpu();
    void update_io();
    void update_comm();
    // write time to a file
    void print_result();


  private:
    double IO_time;
    double CPU_time;
    double COMM_time;
    double time_point_IO;
    double time_point_CPU;
    double time_point_COMM;
};