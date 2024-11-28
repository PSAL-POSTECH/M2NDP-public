BINDIR ?= `pwd`
CPPSRC = ./graph_parser/util.cpp ./graph_parser/parse.cpp
CUSRC =
BASEEXE = pagerank
VARIANT ?= SPMV
# CUDA source files (compiled with cudacc)
ifeq ($(VARIANT),DEFAULT)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE).out
    CUSRC += pagerank.cu
else ifeq ($(VARIANT),SPMV)
    EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)_spmv.out
    CUSRC += pagerank_spmv.cu
endif
OPTS = -O3
INC = -I$(CUDA_INSTALL_PATH)/include
LIBS = -lcudart

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

$(OBJS):
	g++ -c $(CPPSRC) $(OPTS)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
