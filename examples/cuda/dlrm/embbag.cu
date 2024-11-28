#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#define FEATSIZE 256
#define FEATSIZELOG2 8
#define MAX_BATCHES 256

__global__ void EmbeddingBag(
    const int *input, 
    const int *offsets,
    const float *weight,
    float *output,  
    int64_t numBags
    ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bag = (int64_t)idx >> FEATSIZELOG2;  
    int64_t featureDim = (FEATSIZE - 1) & (int64_t)idx;
    
    int64_t begin = offsets[bag];
    int64_t end = offsets[bag+1];
    const float *weightFeat = weight + featureDim;
    float weightFeatSum = 0;
    
    for (int64_t emb = begin; emb < end; emb++) {
        const int64_t weightRow = input[emb];
        float weightValue = weightFeat[weightRow * FEATSIZE];
        weightFeatSum += weightValue;
    }
    output[bag * FEATSIZE + featureDim] = weightFeatSum;
}

int main() {
    const int numBags = 32;  // batchsize 1 4 8 10 20 30 32 256
    const int numLookup = 80;
    const int value_size = numLookup * numBags;
    const int featureSize = FEATSIZE;
    const int numRows = 1000000;

    int input[MAX_BATCHES * numLookup];
    int offsets[numBags + 1];
    
    std::ifstream file("../../data/kaggle_emb/emb_0.txt");
    if (!file.is_open()) {
        std::cerr << "Cannot open the file" << std::endl;
        return 1;
    }

    int number;
    int idx = 0;
    std::string line;
    for (int b_idx=0; b_idx < MAX_BATCHES; b_idx++){
        std::getline(file, line);
        std::stringstream ss(line);
        for (int f_idx = 0; f_idx < numLookup; f_idx++) {
            ss >> number;
            input[idx++] = number;
        }
    }

    for (int i=0; i<numBags + 1; i++)
        offsets[i] = numLookup * i;

    int *d_input, *d_offsets;
    float *d_weight, *d_output;
    cudaMalloc(&d_input, value_size * sizeof(int));
    cudaMalloc(&d_offsets, (numBags + 1) * sizeof(int));
    cudaMalloc(&d_weight, numRows * featureSize * sizeof(float));
    cudaMalloc(&d_output, numBags * featureSize * sizeof(float));

    
    cudaMemcpy(d_offsets, offsets, (numBags + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = numBags * featureSize / blockSize;
    printf("GRID = %d, BLOCK = %d\n", gridSize, blockSize);
    for(int i = 0; i < MAX_BATCHES / numBags; i++) {
        cudaMemcpy(d_input, input + i * value_size, value_size * sizeof(int), cudaMemcpyHostToDevice);
        EmbeddingBag<<<gridSize, blockSize>>>(
        d_input, d_offsets, d_weight, d_output, numBags);
    }
    

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    cudaFree(d_input);
    cudaFree(d_offsets);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}