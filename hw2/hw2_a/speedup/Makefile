CC = mpicc
CXX = mpicxx
LDLIBS = -lpng
CFLAGS = -lm -O3 -ffast-math -msse2
hw2a: CFLAGS += -pthread
hw2b: -cc=clang
hw2b: -cxx=clang++
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)