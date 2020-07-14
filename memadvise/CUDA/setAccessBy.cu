
#include <cuda.h>
#include <stdio.h>

__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}
__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}
int main() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId); 

  write<<< 1, 1000 >>>(ret, 10, 100);            
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
    printf("%d: A+B = %d\n", i, ret[i]);       

  append<<< 1, 1000 >>>(ret, 10, 100);            
  cudaDeviceSynchronize();                        
  cudaFree(ret);
  return 0;
}
