#ifndef PAIRWISE_SQR_DIST_H
#define PAIRWISE_SQR_DIST_H

#define BLOCK_DIM 16
#define CEIL_DIV(a, b) (a + b - 1) / b

void pairwiseSquaredDist(
    cublasHandle_t,
    float*, float*,
    float*, float*,
    int, int, int,
    float*, float*, float*,
    dim3*, dim3*,
    float*
);

__global__ void addSquaredNorms(float*, float*, float*, int, int);

__global__ void argMin(float*, int, int, int*);

__global__ void makeAvgMatrix(float*, int*, int*, int, int);

#endif
