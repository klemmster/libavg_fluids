#ifndef CUDACARD_H_
#define CUDACARD_H_

#define AVG_PLUGIN

#include <api.h>

#include <GL/gl.h>

#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>

// includes, project
#include <sdkHelper.h>  // helper for shared functions common to CUDA SDK samples
#include <shrQATest.h>

#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
   if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			   file, line, (int)err, cudaGetErrorString( err ) );
	   exit(-1);
   }
}

namespace avg
{

class CudaDevice {

public:
    /**
     * Initializes the first CudaCard found on the system
     */
	CudaDevice();
	virtual ~CudaDevice();
    /**
     * @return NVIDIA's CUDevice handle
     */
	CUdevice * getDevice();

private:
	CUdevice * m_cuDevice;
    CUcontext * m_cuContext;
};

} /* avg */

#endif /* CUDACARD_H_ */
