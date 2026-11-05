#include <cuda_runtime.h>
#include <cstdio>
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("SM %d.%d\n", p.major, p.minor);
  printf("sharedMemPerBlock=%zu\n", p.sharedMemPerBlock);
  printf("sharedMemPerBlockOptin=%zu (%d KB)\n", p.sharedMemPerBlockOptin, (int)(p.sharedMemPerBlockOptin/1024));
  printf("sharedMemPerMultiprocessor=%zu\n", p.sharedMemPerMultiprocessor);
  return 0;
}
