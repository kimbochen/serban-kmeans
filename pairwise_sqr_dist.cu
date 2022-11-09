#include <assert.h>
#include <limits.h>
#include <stdio.h>

#include <cublas_v2.h>
#include <cuda.h>

#include "pairwise_sqr_dist.h"


void pairwiseSquaredDist(
    cublasHandle_t handle,
    float *q, float *r,
    float *sqr_norm_q, float *sqr_norm_r,
    int n_q, int n_r, int n_d,
    float *coeff_p, float *zero_p, float *one_p,
    dim3 *grid_p, dim3 *block_p,
    float *pw_sqr_dist
)
{
    /*
     * Computes pairwise squared distance between 2 sets of vectors.
     * Inputs:
     *   q          : 1st set of vectors.        [n_d, n_q]
     *   r          : 2nd set of vectors.        [n_d, n_r]
     *   pw_sqr_dist: Pairwise squared distance. [n_q, n_r]
     *   n_q: Number of vectors in q.
     *   n_r: Number of vectors in r.
     *   n_d: Number of dimensions in each vector.
     */

    // Compute squared norms
    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        1, 1, n_d,              /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        one_p,                    /* Scalar multiplied before A */
        q, n_d, n_d,             /* Matrix A, its leading dimension, and its stride */
        q, n_d, n_d,             /* Matrix B, its leading dimension, and its stride */
        zero_p,                   /* Scalar multiplied before C */
        sqr_norm_q, 1, 1,         /* Matrix C, its leading dimension, and its stride */
        n_q
    );

    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        1, 1, n_d,              /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        one_p,                    /* Scalar multiplied before A */
        r, n_d, n_d,             /* Matrix A, its leading dimension, and its stride */
        r, n_d, n_d,             /* Matrix B, its leading dimension, and its stride */
        zero_p,                   /* Scalar multiplied before C */
        sqr_norm_r, 1, 1,         /* Matrix C, its leading dimension, and its stride */
        n_r
    );

    // Compute the 3rd term for pairwise squared distance
    cublasSgemm(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        n_q, n_r, n_d,            /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        coeff_p,                  /* Scalar multiplied before A */
        q, n_d,                   /* Matrix A and its leading dimension */
        r, n_d,                   /* Matrix B and its leading dimension */
        zero_p,                   /* Scalar multiplied before C */
        pw_sqr_dist, n_q          /* Matrix C and its leading dimension */
    );

    // Add squared norms to the 3rd term
    addSquaredNorms<<<*grid_p, *block_p>>>(pw_sqr_dist, sqr_norm_q, sqr_norm_r, n_q, n_r);
}


__global__ void addSquaredNorms(
    float *pw_sqr_dist, float *sqr_norm_q, float *sqr_norm_r, int n_q, int n_r
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_q && j < n_r) {
        pw_sqr_dist[i + j * n_q] += sqr_norm_q[i] + sqr_norm_r[j];
    }
}


__global__ void argMin(float *arr, int n_rows, int n_cols, int *idxs)
{
    // Reduce along axis 0: [n_rows x n_cols] -> [n_cols]
    int tid = threadIdx.x;
    int j = blockIdx.x;

    if (tid == 0) {
        int min_val = INT_MAX;

        for (int i = 0; i < n_rows; i++) {
            if (arr[i + j * n_rows] < min_val) {
                idxs[j] = i;
                min_val = arr[i + j * n_rows];
            }
        }
    }
}


__global__ void makeAvgMatrix(float *avg, int *membership, int *cluster_sizes, int n_r, int n_q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_r && j < n_q) {
        if (j == membership[i]) {
            avg[i + j * n_r] = 1.0 / cluster_sizes[j];
        }
        else {
            avg[i + j * n_r] = 0.0;
        }
    }
}


/*
int main(void)
{
#define D 2
#define N 3
#define M 15
    float Q[D * N] = {
        80, 70,
        60, 70,
        80, 90
    };
    float R[D * M] = {
        80, 70, 60, 70, 80, 90, 79, 69, 59, 69, 79, 89, 81, 71, 61, 71, 81,
        91, 78, 68, 58, 68, 78, 88, 82, 72, 62, 72, 82, 92
    };
    float pw_sqr_dist[N * M];




    puts("Q:");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.3f ", Q[i + j * D]);  // Stored in column major
        }
        puts("");
    }
    puts("");

    puts("R:");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < M; j++) {
            printf("%.3f ", R[i + j * D]);  // Stored in column major
        }
        puts("");
    }
    puts("");


    float *Q_dev, *R_dev;
    int Q_data_size = D * N * sizeof(float);
    int R_data_size = D * M * sizeof(float);

    cudaMalloc((void**)&Q_dev, Q_data_size);
    cudaMemcpy(Q_dev, Q, Q_data_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&R_dev, R_data_size);
    cudaMemcpy(R_dev, R, R_data_size, cudaMemcpyHostToDevice);


    float *Q_sqr_norm, *R_sqr_norm;
    float *pw_sqr_dist_dev;
    float coeff = -2.0, one = 1.0, zero = 0.0;

    cudaMalloc((void**)&Q_sqr_norm, N * sizeof(float));
    cudaMalloc((void**)&R_sqr_norm, M * sizeof(float));
    cudaMalloc((void**)&pw_sqr_dist_dev, N * M * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    dim3 grid(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM), 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

    pairwiseSquaredDist(
        handle,
        Q_dev, R_dev,
        Q_sqr_norm, R_sqr_norm,
        N, M, D,
        &coeff, &zero, &one,
        &grid, &block,
        pw_sqr_dist_dev
    );
    cudaMemcpy(pw_sqr_dist, pw_sqr_dist_dev, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    puts("pw_sqr_dist:");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%.3f ", pw_sqr_dist[i + j * N]);  // Stored in column major
        }
        puts("");
    }
    puts("");

    int membership[M];
    int *membership_dev;

    puts("membership:");
    for (int i = 0; i < M; i++) {
        membership[i] = -1;
        printf("%d ", membership[i]);
    }
    puts("");

    cudaMalloc((void**)&membership_dev, M * sizeof(int));

    argMin<<<M, N>>>(pw_sqr_dist_dev, N, M, membership_dev);

    cudaMemcpy(membership, membership_dev, M * sizeof(int), cudaMemcpyDeviceToHost);

    puts("new membership:");
    for (int i = 0; i < M; i++) {
        printf("%d ", membership[i]);
    }
    puts("");


    int cluster_sizes[M];
    int *cluster_sizes_dev;

    for (int i = 0; i < M; i++) {
        cluster_sizes[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        cluster_sizes[membership[i]]++;
    }
    cudaMalloc((void**)&cluster_sizes_dev, M * sizeof(int));
    cudaMemcpy(cluster_sizes_dev, cluster_sizes, M * sizeof(int), cudaMemcpyHostToDevice);

    float avg[M * N];
    float *avg_dev;

    cudaMalloc((void**)&avg_dev, M * N * sizeof(float));

    dim3 grid1(CEIL_DIV(M, BLOCK_DIM), CEIL_DIV(N, BLOCK_DIM), 1);
    dim3 block1(BLOCK_DIM, BLOCK_DIM, 1);
    makeAvgMatrix<<<grid1, block1>>>(avg_dev, membership_dev, cluster_sizes_dev, M, N);

    cudaMemcpy(avg, avg_dev, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    puts("avg:");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.3f ", avg[i + j * M]);
        }
        puts("");
    }
    puts("");

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, N, M,
        &one,
        R_dev, D,
        avg_dev, M,
        &zero,
        Q_dev, D
    );
    cudaMemcpy(Q, Q_dev, Q_data_size, cudaMemcpyDeviceToHost);
    puts("new Q:");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.3f ", Q[i + j * D]);  // Stored in column major
        }
        puts("");
    }
    puts("");


    cudaFree(Q_dev);
    cudaFree(R_dev);
    cudaFree(Q_sqr_norm);
    cudaFree(R_sqr_norm);
    cudaFree(pw_sqr_dist_dev);
    cudaFree(membership_dev);
    cudaFree(cluster_sizes_dev);
    cudaFree(avg_dev);
    cublasDestroy(handle);
}
*/
