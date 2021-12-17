CC = clang
CXX = clang++
CXXFLAGS = -O3 -fopenmp -march=native
CFLAGS = -O3 -lm -fopenmp -march=native
NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
LDFLAGS  := -lm
EXES     := hw3-3 hw3-2 hw3-1

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?