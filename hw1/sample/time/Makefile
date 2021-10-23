CC = mpicc
CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm
TARGETS = hw1
OBJECTS = calculate_time.cc calculate_time.h
FILE = calculate_time.o hw1.o hw1 calculate_time.h.gch

.PHONY: all
all: hw1.o calculate_time.o
	mpicxx calculate_time.o hw1.o -o $(TARGETS)

hw1.o: hw1.cc
	mpicxx -c hw1.cc

calculate_time.o: calculate_time.cc calculate_time.h
	mpicxx -c $(OBJECTS)

.PHONY: clean
clean:
	rm -f $(FILE)
