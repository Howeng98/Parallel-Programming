CC = clang
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O2 -ffast-math -march=native
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
