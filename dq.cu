#include <iostream>
#include <thrust/system_error.h>
#include <thrust/version.h>
#if THRUST_VERSION < 100600
#include <thrust/system/cuda_error.h>
#else
#include <thrust/system/cuda/error.h>
#endif

void query_all(std::ostream &os)
{
  int device_count = 0;

  cudaError_t error = cudaGetDeviceCount(&device_count);
  if(error) throw thrust::system_error(error, thrust::cuda_category());

  switch(device_count)
  {
    case 0:
    {
      os << "There is no device supporting CUDA." << std::endl;
      break;
    }

    case 1:
    {
      os << "There is 1 device supporting CUDA." << std::endl;
      break;
    }

    default:
    {
      os << "There are " << device_count << " devices supporting CUDA." << std::endl;
      break;
    }
  }

  for(int dev = 0; dev < device_count; ++dev)
  {
    cudaDeviceProp device_prop;
    cudaError_t error = cudaGetDeviceProperties(&device_prop, dev);
    if(error) throw thrust::system_error(error, thrust::cuda_category());
    
    os << std::endl;
    os << "Device " << dev << ": \"" << device_prop.name << "\"" << std::endl;
    os << "  Major revision number:                         " << device_prop.major << std::endl;
    os << "  Minor revision number:                         " << device_prop.minor << std::endl;
    os << "  Total amount of global memory:                 " << device_prop.totalGlobalMem << " bytes" << std::endl;
    os << "  Number of multiprocessors:                     " << device_prop.multiProcessorCount << std::endl;
    os << "  Total amount of constant memory:               " << device_prop.totalConstMem << " bytes" << std::endl;
    os << "  Total amount of shared memory per block:       " << device_prop.sharedMemPerBlock << " bytes" << std::endl;
    os << "  Total number of registers available per block: " << device_prop.regsPerBlock << std::endl;
    os << "  Warp size:                                     " << device_prop.warpSize << std::endl;
    os << "  Maximum number of threads per block:           " << device_prop.maxThreadsPerBlock << std::endl;
    os << "  Maximum sizes of each dimension of a block:    " << device_prop.maxThreadsDim[0] << " " << device_prop.maxThreadsDim[1] << " " << device_prop.maxThreadsDim[2] << std::endl;
    os << "  Maximum sizes of each dimension of a grid:     " << device_prop.maxGridSize[0] << " " << device_prop.maxGridSize[1] << " " << device_prop.maxGridSize[2] << std::endl;
    os << "  Maximum memory pitch:                          " << device_prop.memPitch << std::endl;
    os << "  Texture alignment:                             " << device_prop.textureAlignment << " bytes" << std::endl;
    os << "  Clock rate:                                    " << device_prop.clockRate * 1e-6f << " GHz" << std::endl;
    os << "  Concurrent copy and execution:                 " << (device_prop.deviceOverlap ? "Yes" : "No") << std::endl;
  }
}

int main()
{
  try
  {
    query_all(std::cout);
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "main(): caught exception: " << e.what() << std::endl;
  }

  return 0;
}

