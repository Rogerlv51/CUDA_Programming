/*
*hello_world.cu   使用nvcc hello_world.cu编译
*/
#include<stdio.h>
#include <unistd.h>

// __global__是告诉编译器这个是个可以在设备上执行的核函数
__global__ void hello_world(void)
{
  printf("GPU: Hello world!\n");
}
int main(int argc, char **argv)
{
  printf("CPU: Hello world!\n");
  sleep(1);
  hello_world<<<1,10>>>();    // 会打印10次
  fflush(stdout);
  printf("cpu: hello\n");  // 可以看到只要核函数运行了，CPU会立马接管主机线程
  cudaDeviceReset();//if no this line ,it can not output hello world from gpu
  /*
    GPU和CPU执行程序是异步的，核函数调用后立刻回到主机线程继续，而不管GPU端核函数是否执行完毕，
    所以上面的程序就是GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程
  */
  return 0;
}

/*
一般CUDA程序分成下面这些步骤：

1、分配GPU内存
2、拷贝内存到设备
3、调用CUDA内核函数来执行计算
4、把计算完成数据拷贝回主机端
5、内存销毁

*/