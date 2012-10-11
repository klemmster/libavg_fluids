#ifndef FLUIDKERNELS_H
#define FLUIDKERNELS_H
#include <cuda.h>

typedef texture< float, cudaTextureType2D, cudaReadModeElementType > fluidTexture;
typedef surface< void, cudaSurfaceType2D > fluidSurface;

fluidTexture frontTex;
fluidSurface backSurface;

fluidTexture veloXTex;
fluidSurface veloXSurface;

fluidTexture veloYTex;
fluidSurface veloYSurface;

fluidTexture tmp1Texture;
fluidSurface tmp1Surface;
fluidTexture tmp2Texture;
fluidSurface tmp2Surface;
fluidTexture divTexture;
fluidSurface divSurface;

extern "C"
__global__ void diffuse_Kernel(float diff, int width, int height, float dt);

extern "C"
__global__ void advect_Kernel(int width, int height, float dt);

extern "C"
__global__ void project_Kernel(int width, int height, float H);

extern "C"
__global__ void project_Kernel2(int width, int height);

extern "C"
__global__ void project_Kernel3(int width, int height, float H);

extern "C"
__global__ void get_back_data_Kernel(float *g_data, int width);

extern "C"
__global__ void get_front_data_Kernel(float *g_data, int width, int height);

extern "C"
__global__ void get_display_Kernel(float *g_data, int widthFac, float pressureContrast,float veloContrast, int width, int height);

extern "C"
__device__ float cuGetCoord(int pos, int stride);

extern "C"
__global__ void testPBO(float *ptr);

#endif //end FluidKernels

