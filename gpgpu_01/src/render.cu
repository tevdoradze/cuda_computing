#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__device__ uchar4 heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return uchar4{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return uchar4{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return uchar4{r, 255, 0, 255};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return uchar4{255, b, 0, 255};
  }
}

__device__ uchar4 palette(int x, int N)
{
  uint8_t v = 255 * x / N;
  return {v,v,v,255};
}

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, int n_iterations)
{
  int xv = blockDim.x * blockIdx.x + threadIdx.x;
  int yv = blockDim.y * blockIdx.y + threadIdx.y;

  if (xv >= width || yv >= height)
    return;

  //uchar4*  lineptr = (uchar4*)(buffer + yv * pitch);
  int*  lineptr = (int*)(buffer + yv * pitch);
  
  float scaled_width = (float)(width - 1);
  float scaled_height = (float)(height - 1);

  float x0 = float(xv)*3.5f/scaled_width-2.5f;
  float y0 = float(yv)*2.0f/scaled_height-1.0f;

  float x = 0.0f;
  float y = 0.0f;

  int iteration = 0;
  float q = (x0 * x0 - x0 * 0.5f + 0.0625f) + y0 * y0;
  if (q * (q + (x0 - 0.25f)) < 0.25f * y0 * y0
      || (x0 + 1.0f)* (x0 + 1.0f) + y0 * y0 < 0.0625f)
    iteration = n_iterations;

  while (iteration < n_iterations && (x*x + y*y) < 4.0f)
  {
    float xtemp = x*x - y*y + x0;
    float ytemp = 2.0f*x*y + y0;
    if (x == xtemp && y == ytemp)
    {
      iteration = n_iterations;
      break;
    }
    x = xtemp;
    y = ytemp;
    iteration = iteration + 1;
  }

  lineptr[xv] = iteration;
}

__global__ void myHist(char* buffer, float* lut, int width, int height, size_t pitch, int n_iterations) {
  // filling lut array
  float total = 0;
  for (int y=0; y < height; y++) {
    int* line_it_value = (int*)(buffer + y * pitch);
    for (int x=0; x < width; x++) {
      int it_value = line_it_value[x];
      lut[it_value] = lut[it_value] + 1.0f;
      if (it_value != n_iterations)
        total += 1.0f;
    }
  }

  // Computing sums / total
  float hue = 0.0f;
  for (int i = 0; i < n_iterations; i++)
  {
    hue += lut[i];
    lut[i] = hue / total;
  }
}

__global__ void applyLut(char* buffer, float* lut, int width, int height, size_t pitch, int n_iterations) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar4*  lineptr = (uchar4*)(buffer + y * pitch);
  uchar4*  lineptr_sym = (uchar4*)(buffer + (height-y-1) * pitch);
  int* line_it_value = (int*)(buffer + y * pitch);
  int it_value = line_it_value[x];
  if (it_value == n_iterations) {
    lineptr[x] = uchar4{0, 0, 0, 255};
    lineptr_sym[x] = uchar4{0, 0, 0, 255};
  }
  else {
    uchar4 c = heat_lut(lut[line_it_value[x]]);
    lineptr[x] = c;
    lineptr_sym[x] = c;  
  }
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

    
    // Run the kernel with blocks of size 64 x 64
    {
      int bsize = 32;
      int w     = std::ceil((float)width / bsize);
      int h     = std::ceil((float)height / bsize);
      
      //Calloc lut
      float*  lut;
      cudaError_t hc = cudaSuccess;

      hc = cudaMalloc(&lut, (n_iterations+1)*sizeof(float));
      if (hc)
        abortError("Fail lut allocation");
      cudaMemset(lut, 0, (n_iterations+1)*sizeof(float));

      spdlog::debug("running kernel of size ({},{})", w, h);

      dim3 dimBlock(bsize, bsize);
      dim3 dimGrid(w, h/2);
      mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations);
      myHist<<<1,1>>>(devBuffer, lut, width, height, pitch, n_iterations);
      applyLut<<<dimGrid, dimBlock>>>(devBuffer, lut, width, height, pitch, n_iterations);
      if (cudaPeekAtLastError())
        abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}
