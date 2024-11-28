BINDIR = `pwd`
CPPSRC = ./graph_parser/util.cpp ./graph_parser/parse.cpp
CUSRC =
BASEEXE = sssp
VARIANT ?= CSR
# CUDA source files (compiled with cudacc)
EXECUTABLE = $(BINDIR)/$(BASEEXE)
CUSRC += sssp_csr.cu

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
