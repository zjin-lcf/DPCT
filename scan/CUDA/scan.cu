#include <cuda.h>
#include <assert.h>
#include <stdio.h>

// work-group size * 2
#define N 512


  template<typename dataType>
__global__ void prescan(dataType *g_odata, dataType *g_idata, int n)
{
  __shared__ dataType temp[N];
  int thid = threadIdx.x; 
  int offset = 1;
  temp[2*thid]   = g_idata[2*thid];
  temp[2*thid+1] = g_idata[2*thid+1];
  for (int d = n >> 1; d > 0; d >>= 1) 
  {
    __syncthreads();
    if (thid < d) 
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid == 0) temp[n-1] = 0; // clear the last elem
  for (int d = 1; d < n; d *= 2) // traverse down
  {
    offset >>= 1;     
    __syncthreads();      
    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  g_odata[2*thid] = temp[2*thid];
  g_odata[2*thid+1] = temp[2*thid+1];
}

template <typename dataType>
void runTest (dataType *in, dataType *out, int n) 
{
  dataType *d_in;
  dataType *d_out;
  cudaMalloc((void**)&d_in, N*sizeof(dataType));
  cudaMalloc((void**)&d_out, N*sizeof(dataType));
  cudaMemcpy(d_in, in, N*sizeof(dataType), cudaMemcpyHostToDevice);
  for (int i = 0; i < 100; i++)
    prescan<<<1, N/2>>>(d_out, d_in, n);
  cudaMemcpy(out, d_out, N*sizeof(dataType), cudaMemcpyDeviceToHost);
}

int main() 
{
  float in[N];
  float gpu_out[N];

  for (int i = 0; i < N; i++) in[i] = (i % 5)+1;

  runTest(in, gpu_out, N);

  return 0;
}
