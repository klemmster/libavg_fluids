/*
 * CudaCard.cpp
 *
 *  Created on: May 30, 2012
 *      Author: andyf
 */

#include "CudaDevice.h"

namespace avg
{


CudaDevice::CudaDevice() {

	CUresult error;

	error = cuInit(0);
	if (error != CUDA_SUCCESS){
		printf("cuda init error");
	}
	int deviceCount = 0;
	error = cuDeviceGetCount(&deviceCount);
	if (error != CUDA_SUCCESS){
		throw std::runtime_error("Failed to get CudaDeviceCount.\n");
	}

	if (deviceCount == 0) {
		throw std::runtime_error("There is no device supporting CUDA.\n");
	}
	int devID=0;
	m_cuDevice = new CUdevice();
	error = cuDeviceGet(m_cuDevice, devID);
	if (error != CUDA_SUCCESS){
		throw std::runtime_error("could not get device\n");
	}
    error = cuGLCtxCreate(m_cuContext, CU_CTX_SCHED_AUTO, *m_cuDevice);
	if (error != CUDA_SUCCESS){
        std::cout << "Error: " << error << "\n";
		throw std::runtime_error("could not create cudaGLContext\n");
	}
    checkCudaErrors(cudaGLSetGLDevice(*m_cuDevice));
}

CudaDevice::~CudaDevice() {
    std::cout << "Destroy Cuda Device\n";
    CUresult error;
    error = cuCtxDestroy(*m_cuContext);
    if (error != CUDA_SUCCESS){
        std::cout << "Context not properly destroyed\n";
    }
    delete m_cuDevice;
    cudaDeviceReset();
}


CUdevice * CudaDevice::getDevice() {
	return m_cuDevice;
}

} /* avg */
