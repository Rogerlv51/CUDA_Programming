#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

// 这是一个简单的两个向量相加的cuda程序

// cpu加法
void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}

// GPU加法
__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  int i=threadIdx.x;
  res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  int nElem=32;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  float *res_from_gpu_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));
  /*
    分配设备端的内存空间，为了区分设备和主机端内存，我们可以给变量加后缀或者前缀h_表示host，d_表示device
    一个经常会发生的错误就是混用设备和主机的内存地址！！!
  */

  initialData(a_h,nElem);
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  /*
    cudaError_t cudaMemcpy(void * dst,const void * src,size_t count, cudaMemcpyKind kind)
    这个函数是内存拷贝过程，可以完成以下几种过程（cudaMemcpyKind kind），从名称上可以很直观地看出内存拷贝的方向
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
    如果函数执行成功，则会返回 cudaSuccess 否则返回 cudaErrorMemoryAllocation
    可以使用如下函数把错误信息翻译成详细信息：
        char* cudaGetErrorString(cudaError_t error)
  */

  dim3 block(nElem);
  dim3 grid(nElem/block.x);
  sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d);
  printf("Execution configuration<<<%d,%d>>>\n",block.x,grid.x);

  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  sumArrays(a_h,b_h,res_h,nElem);

  checkResult(res_h,res_from_gpu_h,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}