CC = mpicc
CXX = mpicxx
LDLIBS = -lpng
CFLAGS = -lm -O3 -ffast-math -msse2
hw2a: -cc=clang -cxx=clang++ CFLAGS += -pthread
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a

.PHONY: all
all: hw2a.o calculate_time.o
	mpicxx -cxx=clang++ $(LDLIBS) $(CFLAGS) calculate_time.o hw2a.o -o $(TARGETS)

hw2a.o: hw2a.cc
	mpicxx -cxx=clang++ -c hw2a.cc

calculate_time.o: calculate_time.cc calculate_time.h
	mpicxx -cxx=clang++ -c calculate_time.cc calculate_time.h

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
