#ifndef TESTCUDAPBO_H_AFS6XRJM
#define TESTCUDAPBO_H_AFS6XRJM

#define AVG_PLUGIN

#include <api.h>
#include <GL/gl.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

#include "CudaDevice.h"

namespace avg
{

class TestCUDAPBO
{
public:
    TestCUDAPBO ();
    virtual ~TestCUDAPBO ();

    void step();
    void setPBO(unsigned pbo);

private:
    unsigned m_pbo;
    cudaGraphicsResource_t m_cuPBO;
    void *m_devPtr;
    size_t m_size;
};

} /* avg */

#endif /* end of include guard: TESTCUDAPBO_H_AFS6XRJM */

