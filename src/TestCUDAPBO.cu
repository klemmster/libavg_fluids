#include "TestCUDAPBO.h"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <iostream>

extern "C" __global__
void testPBO( char* dst){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= 0 && x < 512)
        if(y >= 0  && y < 512){
        int index = y*512 + x;
            dst[index] = (char)index%254;
        }
}

namespace avg
{

TestCUDAPBO::TestCUDAPBO()
{
}

TestCUDAPBO::~TestCUDAPBO(){

}

void TestCUDAPBO::step(){

    checkCudaErrors(cudaGraphicsMapResources( 1, &m_cuPBO));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&m_devPtr, &m_size, m_cuPBO));

    /*
    std::cout << m_devPtr << "\n";
    std::cout << m_size << "\n";
    */

    size_t blocksW = (size_t)ceilf( 512 / 16.0f);
    size_t blocksH = (size_t)ceilf( 512 / 16.0f);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(blocksW, blocksH, 1);
    testPBO<<< dimGrid, dimBlock >>>((char*)m_devPtr);
    checkCudaErrors(cudaGetLastError());
    cudaGraphicsUnmapResources(1, &m_cuPBO);
}

void TestCUDAPBO::setPBO(unsigned pbo){
    m_pbo = pbo;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer( &m_cuPBO, pbo, cudaGraphicsMapFlagsNone));
}

} /* avg */


