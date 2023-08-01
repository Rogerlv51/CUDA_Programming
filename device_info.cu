#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char** argv)
{
    printf("%s Starting ...\n",argv[0]);
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    // cudaGetDeviceCount返回设备上可用计算GPU数量
    if(error_id!=cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n ->%s\n",
              (int)error_id,cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if(deviceCount==0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    }
    int dev=0,driverVersion=0,runtimeVersion=0;
    cudaSetDevice(dev);   // cudaSetDevice接口指定要使用的设备编号，Valid device id's are 0 to (cudaGetDeviceCount() - 1).
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev); 
    // 返回指定设备号的属性信息并保存在cudaDeviceProp这个结构体指针中，这个结构体有很多成员，具体可以查询官方文档
    printf("Device %d:\"%s\"\n",dev,deviceProp.name);  // 比如这里的name就是ASCII string identifying the device，RTX2060等等
    cudaDriverGetVersion(&driverVersion);  // 返回本地安装的最高CUDA版本号
    cudaRuntimeGetVersion(&runtimeVersion); // 返回当前正在使用的CUDA版本号
    printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
        driverVersion/1000,(driverVersion%100)/10,
        runtimeVersion/1000,(runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
        deviceProp.major,deviceProp.minor);  // major和minor定义设备的计算能力大小
    printf("  Total amount of global memory:                %.2f GBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/pow(1024.0,3),deviceProp.totalGlobalMem); // totalGlobalMem返回设备总显存大小
    printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);  // clockRate返回时钟频率
    printf("  Memory Bus width:                             %d-bits\n",
            deviceProp.memoryBusWidth); // 显存总线宽度
    if (deviceProp.l2CacheSize) // L2缓存大小
    {
        printf("  L2 Cache Size:                            	%d bytes\n",
                deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
            ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
            deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory               %lu bytes\n",
            deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:      %lu bytes\n",
            deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block:%d\n",
            deviceProp.regsPerBlock);
    printf("  Wrap size:                                    %d\n",deviceProp.warpSize);
    printf("  Maximun number of thread per multiprocesser:  %d\n",
            deviceProp.maxThreadsPerMultiProcessor); // 多线程最大线程数量
    printf("  Maximun number of thread per block:           %d\n",
            deviceProp.maxThreadsPerBlock);   // 每个block的最大线程数
    printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);  // 块的最大尺寸
    printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
            deviceProp.maxGridSize[0],
	    deviceProp.maxGridSize[1],
	    deviceProp.maxGridSize[2]);    //网格的最大尺寸
    printf("  Maximu memory pitch                           %lu bytes\n",deviceProp.memPitch);
    printf("----------------------------------------------------------\n");
    printf("Number of multiprocessors:                      %d\n", deviceProp.multiProcessorCount);
    printf("Total amount of constant memory:                %4.2f KB\n",
	deviceProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block:        %4.2f KB\n",
     deviceProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block:  %d\n",
    deviceProp.regsPerBlock);
    printf("Warp size                                       %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor:  %d\n",
	deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:     %d\n",
	deviceProp.maxThreadsPerMultiProcessor/32);
    return EXIT_SUCCESS;
   
}