#include "FluidKernels.h"
#include <math.h>


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

FluidField::FluidField()
{

    //TODO: Hard coded Size
    width = 512;
    height = 512;
    m_dt = 1.00f;
    createBuffers();
}

FluidField::~FluidField()
{

}

void FluidField::step(){

    //TODO: HArd coded limits
    size_t diffIt = 2;
    size_t diffVeloIt = 20;
    //END TODO

    tmpFields->bind();

    //---PRESSURE---
    //advectPressure();

	for(size_t it=0; it<diffIt; ++it){
      diffusePressure();
    }

    /*
    //--VELOCITY---

    //Advection
    advectVelocity();
    project();

    //Diffusion
    for(size_t it=0; it<diffVeloIt; ++it){
        diffuseVelocity();
    }
    project();
    */

    pressureField->bind();
    checkCudaErrors(cudaGraphicsMapResources( 1, &m_cuPBO));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&m_devPtr, &m_size, m_cuPBO));

    size_t blocksW = (size_t)ceilf( 512 / 16.0f);
    size_t blocksH = (size_t)ceilf( 512 / 16.0f);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(blocksW, blocksH, 1);
    //testPBO<<< dimGrid, dimBlock >>>((char*)m_devPtr);
    fill_vbo_i8<<< dimGrid, dimBlock >>>((char*)m_devPtr, 512, 512);
    checkCudaErrors(cudaGetLastError());
    cudaGraphicsUnmapResources(1, &m_cuPBO);
}

void FluidField::createBuffers(){
    m_dimBlock = dim3(16, 16, 1);
    m_dimGrid = dim3(width / m_dimBlock.x, height / m_dimBlock.y, 1);
	//m_dimDisplayGrid = dim3(width*2 / m_dimBlock.x, height / m_dimBlock.y, 1);
    veloFieldX = new CudaBuffer(width, height, CudaBuffer::VELO_X);
    veloFieldY = new CudaBuffer(width, height, CudaBuffer::VELO_Y);
    pressureField = new CudaBuffer(width, height);
	tmpFields = new TmpFields(width, height);
	pressureField->beginUpdate(CudaBuffer::FRONT, CudaBuffer::FRONT);
    srand(2345243);
    for(size_t i=0; i<512; ++i){
        for(size_t j=0; j<512; ++j){
        pressureField->setValue(i, j, (float)rand()/RAND_MAX);
        }
    }
    pressureField->endUpdate(true);
}

void FluidField::diffuseVelocity(){

    //TODO: Temporary Parameter
    float diffVelo = 10.0f;
    // END TODO
    veloFieldX->bind();
        diffuse_Kernel<<< m_dimGrid, m_dimBlock >>>(diffVelo, width, height, m_dt);
        checkCudaErrors(cudaGetLastError());
    veloFieldX->flip();

    veloFieldY->bind();
        diffuse_Kernel<<< m_dimGrid, m_dimBlock >>>(diffVelo, width, height, m_dt);
        checkCudaErrors(cudaGetLastError());
    veloFieldY->flip();

	//diffuseComputationTime += stopTimeMeasurement();
}

void FluidField::diffusePressure(bool profiling){

    //TODO: Hard coded diffusion param
    float diff = 0.001f;
    //END TODO

    pressureField->bind();
	//if(profiling)
		//startProfiling();
        diffuse_Kernel<<< m_dimGrid, m_dimBlock >>>(diff, width, height, m_dt);
        checkCudaErrors(cudaGetLastError());
	//if(profiling)
		//stopProfiling();
    pressureField->flip();

	//diffuseComputationTime += stopTimeMeasurement();
}

void FluidField::advectVelocity(){


    veloFieldX->bind();
        advect_Kernel<<< m_dimGrid, m_dimBlock >>>(width, height, m_dt);
        checkCudaErrors(cudaGetLastError());
    veloFieldX->flip();

    veloFieldY->bind();
        advect_Kernel<<< m_dimGrid, m_dimBlock >>>(width, height, m_dt);
        checkCudaErrors(cudaGetLastError());
    veloFieldY->flip();

    //advectComputationTime += stopTimeMeasurement();
}

void FluidField::advectPressure(){


	//startProfiling();
    pressureField->bind();
    advect_Kernel<<< m_dimGrid, m_dimBlock >>>(width, height, m_dt);
    checkCudaErrors(cudaGetLastError());
    pressureField->flip();
	//stopProfiling();

    //advectComputationTime += stopTimeMeasurement();
}

void FluidField::project(bool profiling){
    veloFieldX->bind(false);
    veloFieldY->bind(false);

    //TODO: hard  coded stuff
    float m_H = 10.0f;
    size_t projectIt = 20;
    //END TODO

	if(profiling)
		//startProfiling();
    project_Kernel<<< m_dimGrid, m_dimBlock >>>(width, height, m_H);
    checkCudaErrors(cudaGetLastError());


	project_Kernel2<<< m_dimGrid, m_dimBlock >>>(width, height);
	tmpFields->flip();
    checkCudaErrors(cudaGetLastError());
	if(profiling)
		//stopProfiling();

        for(size_t i=0; i<projectIt; ++i){
            project_Kernel2<<< m_dimGrid, m_dimBlock >>>(width, height);
			tmpFields->flip();
            checkCudaErrors(cudaGetLastError());
        }

	if(profiling)
		//startProfiling();
        project_Kernel3<<< m_dimGrid, m_dimBlock >>>(width, height, m_H);
        checkCudaErrors(cudaGetLastError());
	if(profiling)
		//stopProfiling();

    veloFieldX->flip();
    veloFieldY->flip();

	//projectComputationTime += stopTimeMeasurement();
}


void FluidField::setPBO(unsigned pbo){
    m_pbo = pbo;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer( &m_cuPBO, pbo, cudaGraphicsMapFlagsNone));
}

} /* avg */
