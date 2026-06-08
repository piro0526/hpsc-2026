#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;
using namespace nvcuda;

// Block tile dimensions
#define BM 128
#define BN 128
#define BK 32
#define NUM_THREADS 256

// Warp layout: 4 along M, 2 along N (8 warps total)
// Each warp handles 32x64 output = 2x4 WMMA 16x16 tiles
#define WARP_ROWS 4
#define WARP_COLS 2
#define WARP_M 32
#define WARP_N 64

// Super-tile size for L2 cache tiling
#define SWIZZLE 4

__global__ void float2half_kernel(half * __restrict__ out,
                                  const float * __restrict__ in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ __launch_bounds__(NUM_THREADS)
void kernel(int dim_m, int dim_n, int dim_k,
            const half * __restrict__ d_a,
            const half * __restrict__ d_b,
            float * __restrict__ d_c) {
  // Swizzled block indices for L2 cache reuse
  const int grid_m = (dim_m + BM - 1) / BM;
  const int grid_n = (dim_n + BN - 1) / BN;
  const int num_super_m = (grid_m + SWIZZLE - 1) / SWIZZLE;
  const int linear_bid = blockIdx.x;
  const int super_size = SWIZZLE * SWIZZLE;
  const int super_id = linear_bid / super_size;
  const int within_id = linear_bid % super_size;
  const int super_x = super_id % num_super_m;
  const int super_y = super_id / num_super_m;
  const int local_x = within_id % SWIZZLE;
  const int local_y = within_id / SWIZZLE;
  const int block_x = super_x * SWIZZLE + local_x;
  const int block_y = super_y * SWIZZLE + local_y;
  if (block_x >= grid_m || block_y >= grid_n) return;

  const int bm = BM * block_x;
  const int bn = BN * block_y;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int warp_row = warp_id / WARP_COLS;  // 0-3
  const int warp_col = warp_id % WARP_COLS;  // 0-1

  // Double-buffered shared memory with padding to avoid bank conflicts
  __shared__ half smem_a[2][BK][BM + 8];
  __shared__ half smem_b[2][BK][BN + 8];

  // Accumulator fragments: 2x4 WMMA tiles per warp
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  #pragma unroll
  for (int r = 0; r < 2; r++)
    #pragma unroll
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  // Load first tile into buffer 0
  // A is column-major: element (m_idx, k_idx) = d_a[k_idx * dim_m + m_idx]
  // Coalesced: consecutive threads access consecutive m values
  #pragma unroll
  for (int i = tid; i < BM * BK; i += NUM_THREADS) {
    int lm = i % BM;
    int lk = i / BM;
    smem_a[0][lk][lm] = d_a[lk * dim_m + bm + lm];
  }
  // B is column-major k x n: element (k_idx, n_idx) = d_b[n_idx * dim_k + k_idx]
  // Coalesced: consecutive threads access consecutive k values
  #pragma unroll
  for (int i = tid; i < BK * BN; i += NUM_THREADS) {
    int lk = i % BK;
    int ln = i / BK;
    smem_b[0][lk][ln] = d_b[(bn + ln) * dim_k + lk];
  }
  __syncthreads();

  for (int ko = 0; ko < dim_k; ko += BK) {
    int cur = (ko / BK) & 1;
    int nxt = 1 - cur;

    // Prefetch next tile into alternate buffer
    if (ko + BK < dim_k) {
      int next_k = ko + BK;
      #pragma unroll
      for (int i = tid; i < BM * BK; i += NUM_THREADS) {
        int lm = i % BM;
        int lk = i / BM;
        smem_a[nxt][lk][lm] = d_a[(next_k + lk) * dim_m + bm + lm];
      }
      #pragma unroll
      for (int i = tid; i < BK * BN; i += NUM_THREADS) {
        int lk = i % BK;
        int ln = i / BK;
        smem_b[nxt][lk][ln] = d_b[(bn + ln) * dim_k + next_k + lk];
      }
    }

    // Compute WMMA on current tile
    int wm_off = warp_row * WARP_M;
    int wn_off = warp_col * WARP_N;

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      // Load A fragments (reused across B columns)
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
      #pragma unroll
      for (int r = 0; r < 2; r++)
        wmma::load_matrix_sync(a_frag[r], &smem_a[cur][kk][wm_off + r * 16], BM + 8);

      #pragma unroll
      for (int c = 0; c < 4; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &smem_b[cur][kk][wn_off + c * 16], BN + 8);
        #pragma unroll
        for (int r = 0; r < 2; r++)
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag, acc[r][c]);
      }
    }

    __syncthreads();
  }

  // Store results to global memory
  #pragma unroll
  for (int r = 0; r < 2; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      int cm = bm + warp_row * WARP_M + r * 16;
      int cn = bn + warp_col * WARP_N + c * 16;
      if (cm + 16 <= dim_m && cn + 16 <= dim_n)
        wmma::store_matrix_sync(&d_c[cn * dim_m + cm], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;

  // Pre-convert to half precision for efficient tensor core loading
  half *A_half, *B_half;
  cudaMalloc(&A_half, m * k * sizeof(half));
  cudaMalloc(&B_half, k * n * sizeof(half));
  float2half_kernel<<<(m * k + 255) / 256, 256>>>(A_half, A, m * k);
  float2half_kernel<<<(k * n + 255) / 256, 256>>>(B_half, B, k * n);
  cudaDeviceSynchronize();

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 m,
                 n,
                 k,
                 &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  // Launch optimized kernel
  int grid_m = (m + BM - 1) / BM;
  int grid_n = (n + BN - 1) / BN;
  int num_super_m = (grid_m + SWIZZLE - 1) / SWIZZLE;
  int num_super_n = (grid_n + SWIZZLE - 1) / SWIZZLE;
  int total_blocks = num_super_m * num_super_n * SWIZZLE * SWIZZLE;
  dim3 grid(total_blocks);
  dim3 block(NUM_THREADS);

  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m, n, k, A_half, B_half, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cudaFree(A_half);
  cudaFree(B_half);
  cublasDestroy(cublas_handle);
}
