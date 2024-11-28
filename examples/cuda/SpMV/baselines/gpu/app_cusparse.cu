
#include <assert.h>
#include <cusparse.h>  // cusparseSpMV
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../support/matrix.h"
#include "../../support/params.h"
#include "../../support/timer.h"
#include "../../support/utils.h"

__global__ void spmv_kernel(CSRMatrix csrMatrix, float* inVector,
                            float* outVector) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < csrMatrix.numRows) {
    float sum = 0.0f;
    for (unsigned int i = csrMatrix.rowPtrs[row];
         i < csrMatrix.rowPtrs[row + 1]; ++i) {
      struct Nonzero nonzero = csrMatrix.nonzeros[i];
      sum += inVector[nonzero.col] * nonzero.value;
    }
    outVector[row] = sum;
  }
}

void storeCSRMatrix(const char* fileName, CSRMatrix csrMatrix, float* inVector,
                    float* outVector) {
  FILE* file = fopen(fileName, "w");
  fprintf(file, "%u %u %u\n", csrMatrix.numRows, csrMatrix.numCols,
          csrMatrix.numNonzeros);
  fprintf(file, "rows\n");
  for (uint32_t i = 0; i < csrMatrix.numRows + 1; ++i) {
    fprintf(file, "%u ", csrMatrix.rowPtrs[i]);
  }
  fprintf(file, "\n");
  fprintf(file, "cols\n");
  for (uint32_t i = 0; i < csrMatrix.numNonzeros; ++i) {
    fprintf(file, "%u ", csrMatrix.nonzeros[i].col);
  }
  fprintf(file, "\n");
  fprintf(file, "values\n");
  for (uint32_t i = 0; i < csrMatrix.numNonzeros; ++i) {
    fprintf(file, "%f ", csrMatrix.nonzeros[i].value);
  }
  fprintf(file, "\n");
  fprintf(file, "inVector\n");
  for (uint32_t i = 0; i < csrMatrix.numCols; ++i) {
    fprintf(file, "%f ", inVector[i]);
  }
  fprintf(file, "\n");
  fprintf(file, "outVector\n");
  for (uint32_t i = 0; i < csrMatrix.numRows; ++i) {
    fprintf(file, "%f ", outVector[i]);
  }
  fclose(file);
}

int main(int argc, char** argv) {
  // Process parameters
  struct Params p = input_params(argc, argv);

  // Initialize SpMV data structures
  PRINT_INFO(p.verbosity >= 1, "Reading matrix %s", p.fileName);
  struct COOMatrix cooMatrix = readCOOMatrix(p.fileName);
  PRINT_INFO(p.verbosity >= 1, "    %u rows, %u columns, %u nonzeros",
             cooMatrix.numRows, cooMatrix.numCols, cooMatrix.numNonzeros);
  struct CSRMatrix csrMatrix = coo2csr(cooMatrix);
  uint32_t* row_ptrs =
      (uint32_t*)malloc((csrMatrix.numRows + 1) * sizeof(uint32_t));
  for (int i = 0; i < csrMatrix.numRows + 1; i++) {
    row_ptrs[i] = csrMatrix.rowPtrs[i];
  }
  uint32_t* col_ptrs =
      (uint32_t*)malloc((csrMatrix.numNonzeros) * sizeof(uint32_t));
  float* values = (float*)malloc((csrMatrix.numNonzeros) * sizeof(float));
  for (int i = 0; i < csrMatrix.numNonzeros; i++) {
    col_ptrs[i] = csrMatrix.nonzeros[i].col;
    values[i] = csrMatrix.nonzeros[i].value;
  }
  float* inVector = (float*)malloc(csrMatrix.numCols * sizeof(float));
  float* outVector = (float*)malloc(csrMatrix.numRows * sizeof(float));
  initVector(inVector, csrMatrix.numCols);

  // Allocate data structures on GPU
  uint32_t* row_ptrs_d;
  cudaMalloc((void**)&row_ptrs_d, (csrMatrix.numRows + 1) * sizeof(uint32_t));
  cudaMemcpy(row_ptrs_d, row_ptrs, (csrMatrix.numRows + 1) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  uint32_t* col_ptrs_d;
  cudaMalloc((void**)&col_ptrs_d, csrMatrix.numNonzeros * sizeof(uint32_t));
  cudaMemcpy(col_ptrs_d, col_ptrs, csrMatrix.numNonzeros * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  float* values_d;
  cudaMalloc((void**)&values_d, csrMatrix.numNonzeros * sizeof(float));
  cudaMemcpy(values_d, values, csrMatrix.numNonzeros * sizeof(float),
             cudaMemcpyHostToDevice);
  float alpha = 1.0f;
  float beta = 0.0f;
  float* inVector_d;
  cudaMalloc((void**)&inVector_d, csrMatrix.numCols * sizeof(float));
  float* outVector_d;
  cudaMalloc((void**)&outVector_d, csrMatrix.numRows * sizeof(float));
  // Copy data to GPU
  cudaMemcpy(inVector_d, inVector, csrMatrix.numCols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  cusparseCreate(&handle);
  cusparseCreateCsr(&matA, csrMatrix.numRows, csrMatrix.numCols,
                    csrMatrix.numNonzeros, row_ptrs_d,
                    col_ptrs_d, values_d, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&vecX, csrMatrix.numCols, inVector_d, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, csrMatrix.numRows, outVector_d, CUDA_R_32F);
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_32F,
                          CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  // Calculating result on GPU
  PRINT_INFO(p.verbosity >= 1, "Calculating result on GPU");
  Timer timer;
  startTimer(&timer);
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
  cudaDeviceSynchronize();
  stopTimer(&timer);
  if (p.verbosity == 0) PRINT("%f", getElapsedTime(timer) * 1e3);
  PRINT_INFO(p.verbosity >= 1, "    Elapsed time: %f ms",
             getElapsedTime(timer) * 1e3);

  // Copy data from GPU
  cudaMemcpy(outVector, outVector_d, csrMatrix.numRows * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  storeCSRMatrix("gpu.out", csrMatrix, inVector, outVector);
  // Calculating result on CPU
  PRINT_INFO(p.verbosity >= 1, "Calculating result on CPU");
  float* outVectorReference = (float*)malloc(csrMatrix.numRows * sizeof(float));
  for (uint32_t rowIdx = 0; rowIdx < csrMatrix.numRows; ++rowIdx) {
    float sum = 0.0f;
    for (uint32_t i = csrMatrix.rowPtrs[rowIdx];
         i < csrMatrix.rowPtrs[rowIdx + 1]; ++i) {
      uint32_t colIdx = csrMatrix.nonzeros[i].col;
      float value = csrMatrix.nonzeros[i].value;
      sum += inVector[colIdx] * value;
    }
    outVectorReference[rowIdx] = sum;
  }

  // Verify the result
  PRINT_INFO(p.verbosity >= 1, "Verifying the result");
  for (uint32_t rowIdx = 0; rowIdx < csrMatrix.numRows; ++rowIdx) {
    float diff = (outVectorReference[rowIdx] - outVector[rowIdx]) /
                 outVectorReference[rowIdx];
    const float tolerance = 0.00001;
    if (diff > tolerance || diff < -tolerance) {
      PRINT_ERROR("Mismatch at index %u (CPU result = %f, DPU result = %f)",
                  rowIdx, outVectorReference[rowIdx], outVector[rowIdx]);
    }
  }

  // Deallocate data structures
  freeCOOMatrix(cooMatrix);
  freeCSRMatrix(csrMatrix);
  free(inVector);
  free(outVector);
  free(outVectorReference);

  return 0;
}
