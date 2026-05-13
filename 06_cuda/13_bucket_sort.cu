#include <cstdio>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  bucket[i] = 0;
}

__global__ void count_bucket(int *bucket, int *key, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void scan_bucket(int *bucket, int *offset, int range) {
  int i = threadIdx.x;
  if (i >= range) return;
  for (int j=1; j<range; j<<=1) {
    offset[i] = bucket[i];
    __syncthreads();
    if (i >= j) bucket[i] += offset[i-j];
    __syncthreads();
  }
}

__global__ void fill_key(int *key, int *bucket, int range, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int value = 0;
  for (int j=0; j<range; j++) {
    if (i < bucket[j]) {
      value = j;
      break;
    }
  }
  key[i] = value;
}
 
int main() {
  int n = 50;
  int range = 5;
 
  int *key, *bucket, *offset;
  cudaMallocManaged(&key,    n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
 
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
 
  const int M = 32;

  init_bucket<<<(range+M-1)/M, M>>>(bucket, range);
  cudaDeviceSynchronize();

  count_bucket<<<(n+M-1)/M, M>>>(bucket, key, n);
  cudaDeviceSynchronize();

  scan_bucket<<<1, range>>>(bucket, offset, range);
  cudaDeviceSynchronize();

  fill_key<<<(n+M-1)/M, M>>>(key, bucket, range, n);
  cudaDeviceSynchronize();
 
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
 
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
}
