#ifndef FLUIDKERNELS_H
#define FLUIDKERNELS_H

#define AVG_PLUGIN

#include "CudaDevice.h"
#include "CudaBuffer.h"
#include <api.h>

#ifdef __CUDACC__
#include <GL/gl.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

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
    __global__ void testPBO(char *ptr);

    extern "C"
    __global__ void fill_vbo_i8(char *dst, int width, int height);

#endif

namespace avg
{

class FluidField
{
public:
    FluidField ();
    virtual ~FluidField ();

    void step();
    void setPBO(unsigned pbo);

private:

    void createBuffers();
    void advectVelocity();
    void advectPressure();
    void diffusePressure(bool profiling = false);
    void diffuseVelocity();
    void project(bool profiling = false);

    unsigned m_pbo;
    cudaGraphicsResource_t m_cuPBO;
    void *m_devPtr;
    size_t m_size;
    size_t width;
    size_t height;
    dim3 m_dimBlock;
    dim3 m_dimGrid;
    CudaBuffer *veloFieldX;
    CudaBuffer *veloFieldY;
    CudaBuffer *pressureField;
    TmpFields *tmpFields;
    size_t m_dt;
};

} /* avg */

#endif //end FluidKernels

