/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "./graph_parser/parse.h"
#include "./graph_parser/util.h"
#include "kernel_spmv.cu"
#include <cuda_profiler_api.h>
#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

// Iteration count
#define ITER 1

void print_vectorf(float *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;
    char file_name[100] = "\0";

    cudaError_t err = cudaSuccess;

    if (argc == 3) {
        tmpchar = argv[1]; // Graph inputfile
        file_format = atoi(argv[2]);
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Allocate the csr structure
    csr_array *csr;

    // Parse graph files into csr structure
    if (file_format == 1) {
       csr = parseMetis_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else if (file_format == 0) {
       csr = parseCOO_transpose(tmpchar, &num_nodes, &num_edges, directed);
    } else {
       printf("reserve for future");
       exit(1);
    }

    FILE *kernel1_fp = fopen("kernel1_input.txt", "w");
    FILE *kernel2_fp0 = fopen("kernel2_input0.txt", "w");
    FILE *kernel2_fp1 = fopen("kernel2_input1.txt", "w");
    FILE *kernel2_fp2 = fopen("kernel2_input2.txt", "w");
    FILE *kernel2_fp3 = fopen("kernel2_input3.txt", "w");
    FILE *kernel2_fp4 = fopen("kernel2_input4.txt", "w");
    FILE *kernel2_fp5 = fopen("kernel2_input5.txt", "w");
    FILE *kernel2_fp6 = fopen("kernel2_input6.txt", "w");
    FILE *kernel2_fp7 = fopen("kernel2_input7.txt", "w");
    FILE *kernel2_fp8 = fopen("kernel2_input8.txt", "w");
    FILE *kernel2_fp9 = fopen("kernel2_input9.txt", "w");
    FILE *kernel2_fp10 = fopen("kernel2_input10.txt", "w");
    FILE *kernel2_fp11 = fopen("kernel2_input11.txt", "w");
    FILE *kernel2_fp12 = fopen("kernel2_input12.txt", "w");
    FILE *kernel2_fp13 = fopen("kernel2_input13.txt", "w");
    FILE *kernel2_fp14 = fopen("kernel2_input14.txt", "w");
    FILE *kernel2_fp15 = fopen("kernel2_input15.txt", "w");
    FILE *kernel2_fp16 = fopen("kernel2_input16.txt", "w");
    FILE *kernel2_fp17 = fopen("kernel2_input17.txt", "w");
    FILE *kernel2_fp18 = fopen("kernel2_input18.txt", "w");
    FILE *kernel2_fp19 = fopen("kernel2_input19.txt", "w");
    
    FILE *kernel3_fp0 = fopen("kernel3_input0.txt", "w");
    FILE *kernel3_fp1 = fopen("kernel3_input1.txt", "w");
    FILE *kernel3_fp2 = fopen("kernel3_input2.txt", "w");
    FILE *kernel3_fp3 = fopen("kernel3_input3.txt", "w");
    FILE *kernel3_fp4 = fopen("kernel3_input4.txt", "w");
    FILE *kernel3_fp5 = fopen("kernel3_input5.txt", "w");
    FILE *kernel3_fp6 = fopen("kernel3_input6.txt", "w");
    FILE *kernel3_fp7 = fopen("kernel3_input7.txt", "w");
    FILE *kernel3_fp8 = fopen("kernel3_input8.txt", "w");
    FILE *kernel3_fp9 = fopen("kernel3_input9.txt", "w");
    FILE *kernel3_fp10 = fopen("kernel3_input10.txt", "w");
    FILE *kernel3_fp11 = fopen("kernel3_input11.txt", "w");
    FILE *kernel3_fp12 = fopen("kernel3_input12.txt", "w");
    FILE *kernel3_fp13 = fopen("kernel3_input13.txt", "w");
    FILE *kernel3_fp14 = fopen("kernel3_input14.txt", "w");
    FILE *kernel3_fp15 = fopen("kernel3_input15.txt", "w");
    FILE *kernel3_fp16 = fopen("kernel3_input16.txt", "w");
    FILE *kernel3_fp17 = fopen("kernel3_input17.txt", "w");
    FILE *kernel3_fp18 = fopen("kernel3_input18.txt", "w");
    FILE *kernel3_fp19 = fopen("kernel3_input19.txt", "w");

    fprintf(kernel1_fp, "row_array\n");
    fprintf(kernel2_fp0, "row_array\n");
    fprintf(kernel2_fp1, "row_array\n");
    fprintf(kernel2_fp2, "row_array\n");
    fprintf(kernel2_fp3, "row_array\n");
    fprintf(kernel2_fp4, "row_array\n");
    fprintf(kernel2_fp5, "row_array\n");
    fprintf(kernel2_fp6, "row_array\n");
    fprintf(kernel2_fp7, "row_array\n");
    fprintf(kernel2_fp8, "row_array\n");
    fprintf(kernel2_fp9, "row_array\n");
    fprintf(kernel2_fp10, "row_array\n");
    fprintf(kernel2_fp11, "row_array\n");
    fprintf(kernel2_fp12, "row_array\n");
    fprintf(kernel2_fp13, "row_array\n");
    fprintf(kernel2_fp14, "row_array\n");
    fprintf(kernel2_fp15, "row_array\n");
    fprintf(kernel2_fp16, "row_array\n");
    fprintf(kernel2_fp17, "row_array\n");
    fprintf(kernel2_fp18, "row_array\n");
    fprintf(kernel2_fp19, "row_array\n");
    for (int i = 0; i < num_nodes + 1; i++){
        fprintf(kernel1_fp, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp0, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp1, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp2, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp3, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp4, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp5, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp6, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp7, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp8, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp9, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp10, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp11, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp12, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp13, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp14, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp15, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp16, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp17, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp18, "%d\n", csr->row_array[i]);
        fprintf(kernel2_fp19, "%d\n", csr->row_array[i]);
    }

    fprintf(kernel1_fp, "col_array\n");
    fprintf(kernel2_fp0, "col_array\n");
    fprintf(kernel2_fp1, "col_array\n");
    fprintf(kernel2_fp2, "col_array\n");
    fprintf(kernel2_fp3, "col_array\n");
    fprintf(kernel2_fp4, "col_array\n");
    fprintf(kernel2_fp5, "col_array\n");
    fprintf(kernel2_fp6, "col_array\n");
    fprintf(kernel2_fp7, "col_array\n");
    fprintf(kernel2_fp8, "col_array\n");
    fprintf(kernel2_fp9, "col_array\n");
    fprintf(kernel2_fp10, "col_array\n");
    fprintf(kernel2_fp11, "col_array\n");
    fprintf(kernel2_fp12, "col_array\n");
    fprintf(kernel2_fp13, "col_array\n");
    fprintf(kernel2_fp14, "col_array\n");
    fprintf(kernel2_fp15, "col_array\n");
    fprintf(kernel2_fp16, "col_array\n");
    fprintf(kernel2_fp17, "col_array\n");
    fprintf(kernel2_fp18, "col_array\n");
    fprintf(kernel2_fp19, "col_array\n");
    for (int i = 0; i < num_edges; i++){
        fprintf(kernel1_fp, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp0, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp1, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp2, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp3, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp4, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp5, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp6, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp7, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp8, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp9, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp10, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp11, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp12, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp13, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp14, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp15, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp16, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp17, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp18, "%d\n", csr->col_array[i]);
        fprintf(kernel2_fp19, "%d\n", csr->col_array[i]);
    }
    fprintf(kernel1_fp, "col_cnt\n");
    for (int i = 0; i < num_nodes; i++)
        fprintf(kernel1_fp, "%d\n", csr->col_cnt[i]);
    fclose(kernel1_fp);
    // Allocate rank_arrays
    float *pagerank_array = (float *)malloc(num_nodes * sizeof(float));
    if (!pagerank_array) fprintf(stderr, "malloc failed page_rank_array\n");
    float *pagerank_array2 = (float *)malloc(num_nodes * sizeof(float));
    if (!pagerank_array2) fprintf(stderr, "malloc failed page_rank_array2\n");
    float *data = (float *)malloc(num_edges * sizeof(float));
    // if (!pagerank_array2) fprintf(stderr, "malloc failed page_rank_array2\n");
    int *row_d;
    int *col_d;
    float *data_d;

    float *pagerank1_d;
    float *pagerank2_d;
    int *col_cnt_d;

    // Create device-side buffers for the graph
    err = cudaMalloc(&row_d, (num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc row_d (size:%d) => %s\n",  num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_d (size:%d) => %s\n",  num_edges, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&data_d, num_edges * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc data_d (size:%d) => %s\n", num_edges, cudaGetErrorString(err));
        return -1;
    }

    // Create buffers for pagerank
    err = cudaMalloc(&pagerank1_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc pagerank1_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&pagerank2_d, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc pagerank2_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_cnt_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_cnt_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    double timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Copy the data to the device-side buffers
    err = cudaMemcpy(row_d, csr->row_array, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR:#endif cudaMemcpy row_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_d, csr->col_array, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_cnt_d, csr->col_cnt, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_cnt_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
        return -1;
    }

    // Set up work dimensions
    int block_size = 64;
    // int num_blocks = (num_nodes + block_size - 1) / block_size;
    int rows_per_block = block_size / 32;
    int num_blocks = (num_nodes + rows_per_block - 1) / rows_per_block;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    double timer3 = gettime();
    cudaProfilerStart();
    // Launch the initialization kernel
    inibuffer <<<grid, threads>>>(pagerank1_d, pagerank2_d, num_nodes);
    err = cudaMemcpy(pagerank_array, pagerank1_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(pagerank_array2, pagerank2_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    FILE *kernel0_fp = fopen("kernel0_output.txt", "w");
    fprintf(kernel0_fp, "pagerank1\n");
    for (int i = 0; i < num_nodes; i++)
        fprintf(kernel0_fp, "%.16f\n", pagerank_array[i]);
    fprintf(kernel0_fp, "pagerank2\n");
    for (int i = 0; i < num_nodes; i++)
        fprintf(kernel0_fp, "%.16f\n", pagerank_array2[i]);
    fclose(kernel0_fp);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

    // Initialize the CSR
    inicsr <<<grid, threads>>>(row_d, col_d, data_d, col_cnt_d, num_nodes,
                               num_edges);
    cudaThreadSynchronize();
    
    err = cudaMemcpy(data, data_d, num_edges * sizeof(float), cudaMemcpyDeviceToHost);
    FILE *kernel1_out = fopen("kernel1_output.txt", "w");
    fprintf(kernel2_fp0, "data_array\n");
    fprintf(kernel2_fp1, "data_array\n");
    fprintf(kernel2_fp2, "data_array\n");
    fprintf(kernel2_fp3, "data_array\n");
    fprintf(kernel2_fp4, "data_array\n");
    fprintf(kernel2_fp5, "data_array\n");
    fprintf(kernel2_fp6, "data_array\n");
    fprintf(kernel2_fp7, "data_array\n");
    fprintf(kernel2_fp8, "data_array\n");
    fprintf(kernel2_fp9, "data_array\n");
    fprintf(kernel2_fp10, "data_array\n");
    fprintf(kernel2_fp11, "data_array\n");
    fprintf(kernel2_fp12, "data_array\n");
    fprintf(kernel2_fp13, "data_array\n");
    fprintf(kernel2_fp14, "data_array\n");
    fprintf(kernel2_fp15, "data_array\n");
    fprintf(kernel2_fp16, "data_array\n");
    fprintf(kernel2_fp17, "data_array\n");
    fprintf(kernel2_fp18, "data_array\n");
    fprintf(kernel2_fp19, "data_array\n");

    for (int i = 0; i < num_edges; i++){
       fprintf(kernel1_out, "%.16f\n", data[i]);
       fprintf(kernel2_fp0, "%.16f\n", data[i]);
       fprintf(kernel2_fp1, "%.16f\n", data[i]);
       fprintf(kernel2_fp2, "%.16f\n", data[i]);
       fprintf(kernel2_fp3, "%.16f\n", data[i]);
       fprintf(kernel2_fp4, "%.16f\n", data[i]);
       fprintf(kernel2_fp5, "%.16f\n", data[i]);
       fprintf(kernel2_fp6, "%.16f\n", data[i]);
       fprintf(kernel2_fp7, "%.16f\n", data[i]);
       fprintf(kernel2_fp8, "%.16f\n", data[i]);
       fprintf(kernel2_fp9, "%.16f\n", data[i]);
       fprintf(kernel2_fp10, "%.16f\n", data[i]);
       fprintf(kernel2_fp11, "%.16f\n", data[i]);
       fprintf(kernel2_fp12, "%.16f\n", data[i]);
       fprintf(kernel2_fp13, "%.16f\n", data[i]);
       fprintf(kernel2_fp14, "%.16f\n", data[i]);
       fprintf(kernel2_fp15, "%.16f\n", data[i]);
       fprintf(kernel2_fp16, "%.16f\n", data[i]);
       fprintf(kernel2_fp17, "%.16f\n", data[i]);
       fprintf(kernel2_fp18, "%.16f\n", data[i]);
       fprintf(kernel2_fp19, "%.16f\n", data[i]);
    }
    fclose(kernel1_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

    // Run PageRank for some iter. TO: convergence determination
    for (int i = 0; i < ITER; i++) {
        err = cudaMemcpy(pagerank_array , pagerank1_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(pagerank_array2, pagerank2_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        if(i == 0){
            fprintf(kernel2_fp0, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp0, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp0, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp0, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 1){
            fprintf(kernel2_fp1, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp1, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp1, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp1, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 2){
            fprintf(kernel2_fp2, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp2, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp2, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp2, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 3){
            fprintf(kernel2_fp3, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp3, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp3, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp3, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 4){
            fprintf(kernel2_fp4, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp4, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp4, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp4, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 5){
            fprintf(kernel2_fp5, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp5, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp5, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp5, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 6){
            fprintf(kernel2_fp6, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp6, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp6, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp6, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 7){
            fprintf(kernel2_fp7, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp7, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp7, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp7, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 8){
            fprintf(kernel2_fp8, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp8, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp8, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp8, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 9){
            fprintf(kernel2_fp9, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp9, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp9, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp9, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 10){
            fprintf(kernel2_fp10, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp10, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp10, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp10, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 11){
            fprintf(kernel2_fp11, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp11, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp11, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp11, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 12){
            fprintf(kernel2_fp12, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp12, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp12, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp12, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 13){
            fprintf(kernel2_fp13, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp13, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp13, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp13, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 14){
            fprintf(kernel2_fp14, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp14, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp14, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp14, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 15){
            fprintf(kernel2_fp15, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp15, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp15, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp15, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 16){
            fprintf(kernel2_fp16, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp16, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp16, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp16, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 17){
            fprintf(kernel2_fp17, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp17, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp17, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp17, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 18){
            fprintf(kernel2_fp18, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp18, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp18, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp18, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 19){
            fprintf(kernel2_fp19, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp19, "%.16f\n", pagerank_array[i]);
            fprintf(kernel2_fp19, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel2_fp19, "%.16f\n", pagerank_array2[i]);
        }
        // Launch pagerank kernel 1
        spmv_csr_scalar_kernel <<<grid, threads>>>(num_nodes, row_d, col_d,
                                                   data_d, pagerank1_d,
                                                   pagerank2_d);
        sprintf(file_name, "kernel2_output%d.txt", i);
        err = cudaMemcpy(pagerank_array , pagerank1_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(pagerank_array2, pagerank2_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        FILE* kernel2_out = fopen(file_name, "w");
        for (int j = 0; j < num_nodes; j++)
            fprintf(kernel2_out, "%.16f\n", pagerank_array2[j]);
        fclose(kernel2_out);

        /* kernel 3*/
        if(i == 0){
            fprintf(kernel3_fp0, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp0, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp0, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp0, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 1){
            fprintf(kernel3_fp1, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp1, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp1, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp1, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 2){
            fprintf(kernel3_fp2, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp2, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp2, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp2, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 3){
            fprintf(kernel3_fp3, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp3, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp3, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp3, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 4){
            fprintf(kernel3_fp4, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp4, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp4, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp4, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 5){
            fprintf(kernel3_fp5, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp5, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp5, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp5, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 6){
            fprintf(kernel3_fp6, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp6, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp6, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp6, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 7){
            fprintf(kernel3_fp7, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp7, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp7, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp7, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 8){
            fprintf(kernel3_fp8, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp8, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp8, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp8, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 9){
            fprintf(kernel3_fp9, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp9, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp9, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp9, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 10){
            fprintf(kernel3_fp10, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp10, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp10, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp10, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 11){
            fprintf(kernel3_fp11, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp11, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp11, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp11, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 12){
            fprintf(kernel3_fp12, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp12, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp12, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp12, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 13){
            fprintf(kernel3_fp13, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp13, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp13, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp13, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 14){
            fprintf(kernel3_fp14, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp14, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp14, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp14, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 15){
            fprintf(kernel3_fp15, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp15, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp15, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp15, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 16){
            fprintf(kernel3_fp16, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp16, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp16, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp16, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 17){
            fprintf(kernel3_fp17, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp17, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp17, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp17, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 18){
            fprintf(kernel3_fp18, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp18, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp18, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp18, "%.16f\n", pagerank_array2[i]);
        }
        if(i == 19){
            fprintf(kernel3_fp19, "pagerank1\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp19, "%.16f\n", pagerank_array[i]);
            fprintf(kernel3_fp19, "pagerank2\n");
            for (int i = 0; i < num_nodes; i++)
                fprintf(kernel3_fp19, "%.16f\n", pagerank_array2[i]);
        }
        // Launch pagerank kernel 2
        pagerank2 <<<grid, threads>>>(pagerank1_d, pagerank2_d, num_nodes);
        sprintf(file_name, "kernel3_output%d.txt", i);
        err = cudaMemcpy(pagerank_array , pagerank1_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        err = cudaMemcpy(pagerank_array2, pagerank2_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        FILE* kernel3_out = fopen(file_name, "w");
        fprintf(kernel3_out, "pagerank1\n");
        for (int j = 0; j < num_nodes; j++)
            fprintf(kernel3_out, "%.16f\n", pagerank_array[j]);
        fprintf(kernel3_out, "pagerank2\n");
        for (int j = 0; j < num_nodes; j++)
            fprintf(kernel3_out, "%.16f\n", pagerank_array2[j]);
        fclose(kernel3_out);
    }
    cudaThreadSynchronize();
    cudaProfilerStop();
    double timer4 = gettime();
    fclose(kernel2_fp0);
    fclose(kernel2_fp1);
    fclose(kernel2_fp2);
    fclose(kernel2_fp3);
    fclose(kernel2_fp4);
    fclose(kernel2_fp5);
    fclose(kernel2_fp6);
    fclose(kernel2_fp7);
    fclose(kernel2_fp8);
    fclose(kernel2_fp9);
    fclose(kernel2_fp10);
    fclose(kernel2_fp11);
    fclose(kernel2_fp12);
    fclose(kernel2_fp13);
    fclose(kernel2_fp14);
    fclose(kernel2_fp15);
    fclose(kernel2_fp16);
    fclose(kernel2_fp17);
    fclose(kernel2_fp18);
    fclose(kernel2_fp19);

    fclose(kernel3_fp0);
    fclose(kernel3_fp1);
    fclose(kernel3_fp2);
    fclose(kernel3_fp3);
    fclose(kernel3_fp4);
    fclose(kernel3_fp5);
    fclose(kernel3_fp6);
    fclose(kernel3_fp7);
    fclose(kernel3_fp8);
    fclose(kernel3_fp9);
    fclose(kernel3_fp10);
    fclose(kernel3_fp11);
    fclose(kernel3_fp12);
    fclose(kernel3_fp13);
    fclose(kernel3_fp14);
    fclose(kernel3_fp15);
    fclose(kernel3_fp16);
    fclose(kernel3_fp17);
    fclose(kernel3_fp18);
    fclose(kernel3_fp19);
    // Copy the rank buffer back
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy() failed (%s)\n", cudaGetErrorString(err));
        return -1;
    }

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    double timer2 = gettime();

    // Report timing characteristics
    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    // Print rank array
    print_vectorf(pagerank_array, num_nodes);
#endif

    // Free the host-side arrays
    free(pagerank_array);
    free(pagerank_array2);
    csr->freeArrays();
    free(csr);

    // Free the device buffers
    cudaFree(row_d);
    cudaFree(col_d);
    cudaFree(data_d);

    cudaFree(pagerank1_d);
    cudaFree(pagerank2_d);

    return 0;

}

void print_vectorf(float *vector, int num)
{
    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++) {
        fprintf(fp, "%f\n", vector[i]);
    }

    fclose(fp);
}

