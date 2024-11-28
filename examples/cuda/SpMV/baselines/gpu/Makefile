all:
	nvcc app.cu -I/usr/local/cuda/include -lm -o spmv
#	nvcc app_cusparse.cu -I/usr/local/cuda/include -lm -o spmv -lcusparse

clean:
	rm spmv
