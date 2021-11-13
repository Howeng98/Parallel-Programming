CC = mpicc
CXX = mpicxx
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: -march=native
hw2a: CC=clang
hw2a: CXX=clang++
hw2a: CFLAGS += -ffast-math
hw2a: CFLAGS += -pthread
hw2b: -cc=clang
hw2b: -cxx=clang++
hw2b: CFLAGS += -march=athlon64
hw2b: CFLAGS += -ffast-math
hw2b: CFLAGS += -fno-associative-math
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)