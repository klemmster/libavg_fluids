#include "CudaBuffer.h"
#include "FluidKernels.h"

#ifdef __CUDACC__
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

extern "C" __global__ void
diffuse_Kernel(float diff, int width, int height, float dt)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float ustep = 1.0f / (float)width;
    float vstep = 1.0f / (float)height;
    float u = cuGetCoord(x, width);
    float v = cuGetCoord(y, height);

    float pC = tex2D(frontTex, u, v);
    float pL = tex2D(frontTex, u-ustep, v);
    float pR = tex2D(frontTex, u+ustep, v);
    float pB = tex2D(frontTex, u, v-vstep);
    float pT = tex2D(frontTex, u, v+vstep);

	float a = dt * diff * width * height;
	float result = (pC + a * (pL + pR + pB + pT))/(1.0f+4.0f*a);
    surf2Dwrite<float>(result, backSurface, x*4, y);
    //surf2Dwrite<float>(pC-0.01f, backSurface, x*4, y);
}

extern "C" __device__ float
cuGetCoord(int pos, int stride){
    return ((pos+0.5f) / stride);
}

extern "C" __global__ void
project_Kernel(int width, int height, float H)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float u = cuGetCoord(x, width);
    float v = cuGetCoord(y, height);
    float ustep = 1.0f / (float)width;
    float vstep = 1.0f / (float)height;

    float rX = tex2D(veloXTex, u+ustep, v);
    float lX = tex2D(veloXTex, u-ustep, v);
    float rY = tex2D(veloYTex, u, v+vstep);
    float lY = tex2D(veloYTex, u, v-vstep);

    float val = -0.5f * H *(rX - lX + rY - lY);

	surf2Dwrite<float>(val, divSurface, x*4, y);
	surf2Dwrite<float>(0.0f, tmp1Surface, x*4, y);
	surf2Dwrite<float>(0.0f, tmp2Surface, x*4, y);
}

extern "C" __global__ void
project_Kernel2(int width, int height){

        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	    float u = cuGetCoord(x, width);
	    float v = cuGetCoord(y, height);
	    float ustep = 1.0f / (float)width;
	    float vstep = 1.0f / (float)height;

        float pL = tex2D(tmp1Texture, u-ustep, v);
        float pR = tex2D(tmp1Texture, u+ustep, v);
        float pT = tex2D(tmp1Texture, u, v-vstep);
        float pB = tex2D(tmp1Texture, u, v+vstep);

        float val = (tex2D(divTexture, u, v) + pL + pR + pT + pB )/4.0f;
		surf2Dwrite<float>(val, tmp2Surface, x*4, y);
}

extern "C" __global__ void
project_Kernel3(int width, int height, float H){

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float u = cuGetCoord(x, width);
    float v = cuGetCoord(y, height);
	float ustep = 1.0f / (float)width;
	float vstep = 1.0f / (float)height;

    float pL = tex2D(tmp1Texture, u-ustep, v);
    float pR = tex2D(tmp1Texture, u+ustep, v);
    float pT = tex2D(tmp1Texture, u, v-vstep);
    float pB = tex2D(tmp1Texture, u, v+vstep);

    float valX = tex2D(veloXTex, u, v) - (0.5f*(pR - pL)/H);
    float valY = tex2D(veloYTex, u, v) - (0.5f*(pB - pT)/H);

    surf2Dwrite<float>(valX, veloXSurface, x*4, y);
    surf2Dwrite<float>(valY, veloYSurface, x*4, y);
}

extern "C" __global__ void
advect_Kernel(int width, int height, float dt)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float u = cuGetCoord(x, width);
    float v = cuGetCoord(y, height);
    float vx = (tex2D(veloXTex, u, v));
    float vy = (tex2D(veloYTex, u, v));
    vx /= width;
    vy /= height;
    surf2Dwrite<float>(tex2D(frontTex, u-vx, v-vy), backSurface, x*4, y);
}

extern "C" __global__ void
get_back_data_Kernel(float * g_data, int width){
     unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
     g_data[y*width+x] = surf2Dread<float>(backSurface, x*4, y);
}

extern "C" __global__ void
get_front_data_Kernel(float * g_data, int width, int height){
     unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
     float u = cuGetCoord(x, width);
     float v = cuGetCoord(y, height);
     g_data[y*width+x] = tex2D(frontTex, u, v);
}

extern "C" __global__ void
fill_vbo_i8(char *dst, int width, int height){
     unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
     float u = cuGetCoord(x, width);
     float v = cuGetCoord(y, height);
     dst[y*width+x] = (char)(tex2D(frontTex, u, v)*255);
}
#endif

CudaBuffer::CudaBuffer(const size_t width, const size_t height, FieldType type):
    m_Width(width),
    m_Height(height),
    m_isUpdating(false),
    m_ReadPtr(0),
    m_WritePtr(0),
    m_dataFront(new float[width*height]),
    m_dataBack(new float[width*height]),
    m_dDataPtr(0),
    m_readPtr(0),
    m_writePtr(0),
    m_dimBlock(dim3(8, 8, 1)),
    m_dimGrid(dim3(width/8, height/8, 1)),
    m_type(type)
{
    size_t size = width*height;
    m_dataSize = size*sizeof(float);
    std::fill(m_dataFront, m_dataFront+(size), 0.0f);
    std::fill(m_dataBack, m_dataBack+(size), 0.0f);

    checkCudaErrors( cudaMalloc( (void**) &m_dDataPtr, m_dataSize));
    m_channelDescr = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    checkCudaErrors( cudaMallocArray( &m_cuFrontPtr, &m_channelDescr, m_Width, m_Height, cudaArraySurfaceLoadStore ));
    checkCudaErrors( cudaMallocArray( &m_cuBackPtr, &m_channelDescr, m_Width, m_Height, cudaArraySurfaceLoadStore ));

    checkCudaErrors( cudaMemcpyToArray( m_cuFrontPtr, 0, 0, m_dataFront, m_dataSize, cudaMemcpyHostToDevice));
    checkCudaErrors( cudaMemcpyToArray( m_cuBackPtr, 0, 0, m_dataBack, m_dataSize, cudaMemcpyHostToDevice));

    frontTex.addressMode[0] = cudaAddressModeWrap;
    frontTex.addressMode[1] = cudaAddressModeWrap;
    frontTex.filterMode = cudaFilterModeLinear;
    frontTex.normalized = true;

    veloXTex.addressMode[0] = cudaAddressModeWrap;
    veloXTex.addressMode[1] = cudaAddressModeWrap;
    veloXTex.filterMode = cudaFilterModeLinear;
    veloXTex.normalized = true;

    veloYTex.addressMode[0] = cudaAddressModeWrap;
    veloYTex.addressMode[1] = cudaAddressModeWrap;
    veloYTex.filterMode = cudaFilterModeLinear;
    veloYTex.normalized = true;

    bind();
}

void CudaBuffer::bind(bool bindFront){

    unbind();
	if(bindFront){
		checkCudaErrors( cudaBindTextureToArray( frontTex, m_cuFrontPtr, m_channelDescr ));
		checkCudaErrors( cudaBindSurfaceToArray( backSurface, m_cuBackPtr ));
	}
    switch(m_type){
        case VELO_X: {
                checkCudaErrors( cudaUnbindTexture( veloXTex) );
                checkCudaErrors( cudaBindTextureToArray( veloXTex, m_cuFrontPtr, m_channelDescr ));
                checkCudaErrors( cudaBindSurfaceToArray( veloXSurface, m_cuBackPtr ));

                break;
                }
        case VELO_Y: {
                checkCudaErrors( cudaUnbindTexture( veloYTex) );
                checkCudaErrors( cudaBindTextureToArray( veloYTex, m_cuFrontPtr, m_channelDescr ));
                checkCudaErrors( cudaBindSurfaceToArray( veloYSurface, m_cuBackPtr ));

                break;
                }
    }
}

void CudaBuffer::unbind(){
	checkCudaErrors( cudaUnbindTexture( frontTex) );
}

CudaBuffer::~CudaBuffer(){
    delete []m_dataFront;
    delete []m_dataBack;

    checkCudaErrors( cudaUnbindTexture( veloXTex) );
    checkCudaErrors( cudaUnbindTexture( veloYTex) );
    checkCudaErrors( cudaUnbindTexture( frontTex) );

    cudaFreeArray(m_cuFrontPtr);
    cudaFreeArray(m_cuBackPtr);
}

float* CudaBuffer::beginUpdate(BufferType readBuffer, BufferType writeBuffer){
    if(! m_isUpdating){
        m_isUpdating = true;
        bind();
        if(readBuffer == CudaBuffer::FRONT){
            get_front_data_Kernel<<< m_dimGrid, m_dimBlock , 0 >>>(m_dDataPtr, m_Width, m_Height);
            checkCudaErrors( cudaMemcpy( m_dataFront, m_dDataPtr, m_dataSize, cudaMemcpyDeviceToHost) );
            m_readPtr = m_dataFront;
        }
        if(readBuffer == CudaBuffer::BACK){
            get_back_data_Kernel<<< m_dimGrid, m_dimBlock , 0 >>>(m_dDataPtr, m_Width);
            checkCudaErrors( cudaMemcpy( m_dataBack, m_dDataPtr, m_dataSize, cudaMemcpyDeviceToHost) );
            m_readPtr = m_dataBack;
        }
        if(writeBuffer == CudaBuffer::FRONT){
            get_front_data_Kernel<<< m_dimGrid, m_dimBlock , 0 >>>(m_dDataPtr, m_Width, m_Height);
            checkCudaErrors( cudaMemcpy( m_dataFront, m_dDataPtr, m_dataSize, cudaMemcpyDeviceToHost) );
            m_writePtr = m_dataFront;
            m_cuWritePtr = m_cuFrontPtr;
        }
        if(writeBuffer == CudaBuffer::BACK){
            get_back_data_Kernel<<< m_dimGrid, m_dimBlock , 0 >>>(m_dDataPtr, m_Width);
            checkCudaErrors( cudaMemcpy( m_dataBack, m_dDataPtr, m_dataSize, cudaMemcpyDeviceToHost) );
            m_writePtr = m_dataBack;
            m_cuWritePtr = m_cuBackPtr;
        }
        if(writeBuffer == CudaBuffer::NONE){
            m_writePtr = 0;
            m_cuWritePtr = 0;
        }
        if(readBuffer == CudaBuffer::NONE) m_readPtr = 0;
        return m_writePtr;
    }else{
        throw UpdatingException("In beginUpdate");
    }
}

void CudaBuffer::flushUpdate(){
    if( m_isUpdating){
        if (m_cuWritePtr){
            checkCudaErrors( cudaMemcpyToArray( m_cuWritePtr, 0, 0, m_writePtr, m_dataSize, cudaMemcpyHostToDevice));
        }else{
            throw UpdatingException("Can't flush to NULL");
        }
    }else{
        throw UpdatingException("In flushUpdate");
    }
}

void CudaBuffer::endUpdate(bool flush){
    if( m_isUpdating){
        if(flush){
         flushUpdate();
        }
        m_cuWritePtr = 0;
        m_writePtr = 0;
        m_readPtr = 0;
        m_isUpdating = false;
    }else{
        throw UpdatingException("In Flush");
    }
}

void CudaBuffer::flip(){
    if(! m_isUpdating){
		checkCudaErrors( cudaUnbindTexture( frontTex ));
        std::swap(m_cuFrontPtr, m_cuBackPtr);
        checkCudaErrors( cudaBindTextureToArray( frontTex, m_cuFrontPtr, m_channelDescr ));
        checkCudaErrors( cudaBindSurfaceToArray( backSurface, m_cuBackPtr));
   }else{
        throw UpdatingException("In flip");
    }
}

void CudaBuffer::setValue(const size_t x, const size_t y, const float value)
{
    if(m_isUpdating){
        setValue(x, y, value, 1.0f);
    }else{
        throw UpdatingException("In Set Value");
    }
}

void CudaBuffer::setValue(const size_t x, const size_t y, const float value, const float weight)
{
    if(m_isUpdating){
        m_writePtr[getIndex(x, y)] = weight*value + (1 - weight)* m_readPtr[getIndex(x, y)] ;
    }else{
        throw UpdatingException("In SetValue");
    }
}


const float& CudaBuffer::operator()(const size_t x, const size_t y) const
{
    return m_readPtr[getIndex(x, y)];

}

const float& CudaBuffer::operator()(const int x, const int y) const
{
    return m_readPtr[getIndex(x, y)];
}

float CudaBuffer::operator()(float x, float y) const
{
    while(x<0)
        x+=m_Width;
    while(x>=m_Width)
        x-=m_Width;
    while(y<0)
        y+=m_Height;
    while(y>=m_Height)
        y-=m_Height;

    float nx = x-(int)x;
    float ny = y-(int)y;

    int iX = (int)x;
    int iY = (int)y;

    if(!m_readPtr){
#ifdef DEBUG_OUTPUT
        std::cout << "No Read Ptr\n";
#endif
    }

    float xInter1 = m_readPtr[getIndex(iX, iY)]*(1-nx) + m_readPtr[getIndex(iX+1, iY)]*nx;
	float xInter2 = m_readPtr[getIndex(iX, iY+1)]*(1-nx) + m_readPtr[getIndex(iX+1, iY+1)]*nx;
	return xInter1*(1-ny)+xInter2*ny;
}


//GLOBAL FIELDS

extern "C" __global__ void
get_display_Kernel(float * g_data, int widthFactor, float pressureContrast,float veloContrast,int width, int height){
     unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x<width) {
		//Pressure display
		float u = cuGetCoord(x-width, width);
    	float v = cuGetCoord(y, height);
		float val = tex2D(frontTex, u, v) * pressureContrast;
		int coord = (y*width*widthFactor+x)*3;
		g_data[coord+2] = val;
		g_data[coord+1] = val;
		g_data[coord] = val;
	}else{
		//Speed display
		float u = cuGetCoord(x-width, width);
    	float v = cuGetCoord(y, height);
		int coord = (y*width*widthFactor+x)*3;
     	g_data[coord] = tex2D(veloXTex, u, v)*0.5f*veloContrast + 0.5f;
		g_data[coord+1] = tex2D(veloYTex, u, v)*0.5f*veloContrast + 0.5f;
		g_data[coord+2] = 0;
	}

}

void TmpFields::flip(){
		checkCudaErrors( cudaUnbindTexture( tmp1Texture) );
		checkCudaErrors( cudaUnbindTexture( tmp2Texture) );
        std::swap(tmp1SurfacePtr, tmp2SurfacePtr);
		checkCudaErrors( cudaBindTextureToArray( tmp1Texture, tmp1SurfacePtr, channelDescr ));
		checkCudaErrors( cudaBindSurfaceToArray( tmp1Surface, tmp1SurfacePtr ));
		checkCudaErrors( cudaBindTextureToArray( tmp2Texture, tmp2SurfacePtr, channelDescr ));
		checkCudaErrors( cudaBindSurfaceToArray( tmp2Surface, tmp2SurfacePtr ));
};

void TmpFields::bind(){

	checkCudaErrors( cudaBindTextureToArray( tmp1Texture, tmp1SurfacePtr, channelDescr ));
	checkCudaErrors( cudaBindSurfaceToArray( tmp1Surface, tmp1SurfacePtr ));

	checkCudaErrors( cudaBindTextureToArray( tmp2Texture, tmp2SurfacePtr, channelDescr ));
	checkCudaErrors( cudaBindSurfaceToArray( tmp2Surface, tmp2SurfacePtr ));

	checkCudaErrors( cudaBindTextureToArray( divTexture, divSurfacePtr, channelDescr ));
	checkCudaErrors( cudaBindSurfaceToArray( divSurface, divSurfacePtr ));
}


void TmpFields::init(int width, int height){

	//Display memory
	m_displayDataSize = width*2*height*sizeof(float)*3;
	checkCudaErrors( cudaMalloc( (void**) &m_displayPtr, m_displayDataSize));

	//Tex'n'Surf
	channelDescr = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	checkCudaErrors( cudaMallocArray( &tmp1SurfacePtr, &channelDescr, width, height, cudaArraySurfaceLoadStore ));
	checkCudaErrors( cudaMallocArray( &tmp2SurfacePtr, &channelDescr, width, height, cudaArraySurfaceLoadStore ));
	checkCudaErrors( cudaMallocArray( &divSurfacePtr, &channelDescr, width, height, cudaArraySurfaceLoadStore ));

	tmp1Texture.addressMode[0] = cudaAddressModeWrap;
	tmp1Texture.addressMode[1] = cudaAddressModeWrap;
	tmp1Texture.filterMode = cudaFilterModePoint;
	tmp1Texture.normalized = true;

	tmp2Texture.addressMode[0] = cudaAddressModeWrap;
	tmp2Texture.addressMode[1] = cudaAddressModeWrap;
	tmp2Texture.filterMode = cudaFilterModePoint;
	tmp2Texture.normalized = true;

	divTexture.addressMode[0] = cudaAddressModeWrap;
	divTexture.addressMode[1] = cudaAddressModeWrap;
	divTexture.filterMode = cudaFilterModePoint;
	divTexture.normalized = true;
}

void TmpFields::free(){
	checkCudaErrors( cudaUnbindTexture( tmp1Texture) );
	checkCudaErrors( cudaUnbindTexture( tmp2Texture) );
	checkCudaErrors( cudaUnbindTexture( divTexture) );
	cudaFreeArray(tmp1SurfacePtr);
	cudaFreeArray(tmp2SurfacePtr);
	cudaFreeArray(divSurfacePtr);
}

