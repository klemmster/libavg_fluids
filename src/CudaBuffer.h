#ifndef CUDABUFFER_H_5I18KWUB
#define CUDABUFFER_H_5I18KWUB

#include <string>
#include <iostream>
#include "Config.h"

using namespace std;

/**
 * Exception indicating that #DoubleBuffer::beginUpdate has been called twice,
 * missing a call to #DoubleBuffer::endUpdate.
 * This would lead to inconsistencies thus, an exception is thrown.
 */
class UpdatingException: public std::exception
{
public:
   /**
     * Use this constructor to give the exception a reason
     * @param cause the reason why this exception is being thrown
     */
    UpdatingException(std::string cause):
        m_cause(cause){};
    virtual ~UpdatingException() throw(){};

    /**
     * @return the reason, why the exception occured
     */
    virtual const char* what() const throw(){
        std::cout << "Updating Exception: " << m_cause << "\n";
        return m_cause.c_str();
    }

private:
     /**
     * Default constructor hidden, we want to ensure there is a reason given.
     */
    UpdatingException(){
        m_cause = "No Reason";
    };

    std::string m_cause;
};

class CudaBuffer{

public:
    /** Specify which texture belongs to this buffer. This is needed, as textures can
     * not be created dynamically, but have to be known in advance.
     */
    enum FieldType{
        PRESSURE, /**< use the pressure texture */
        VELO_X, /**< use the velocity X texture */
        VELO_Y, /**< use the velocity Y texture */
    };

     enum BufferType{
        FRONT, /**< use the front buffer */
        BACK, /**< use the back buffer */
        NONE /**< use no buffer at all */
    };

   /**
     * Creates a buffer of a given size
     * @param width of the field
     * @param height of the field
     * @param type which \link FieldType \endlink should be used
     */
	CudaBuffer (const size_t width, const size_t height, FieldType type=PRESSURE);
    virtual ~CudaBuffer ();
    void flip();

    const float& operator()(const size_t x, const size_t y) const;
    const float& operator()(const int x, const int y) const;
    float operator()(const float x, const float y) const;

    void setValue(const size_t x, const size_t y, const float value);
    void setValue(const size_t x, const size_t y, const float value, const float weight);
    virtual float * beginUpdate(BufferType readBuffer, BufferType writeBuffer);
    virtual void endUpdate(bool flush=true);
    void bind(bool bindFront=true);
    void unbind();

protected:
    virtual size_t getIndex(int x, int y) const
    {
        while(x<0)
    		x += m_Width;
    	while(y<0)
    		y += m_Height;
    	return m_Width*(y%m_Height)+(x%m_Width);
    }

private:

    /** Width of the buffer */
    size_t m_Width;
    /** Height of the buffer */
    size_t m_Height;
    /** True if and only if beginUpdate has been called */
    bool m_isUpdating;
    /** Pointer to the buffer currently read from */
    float* m_ReadPtr;
    /** Pointer to the buffer currently written to. Can be the same buffer as #m_ReadPtr
     */
    float* m_WritePtr;

    cudaChannelFormatDesc m_channelDescr;
    cudaArray* m_cuFrontPtr;
    cudaArray* m_cuBackPtr;
    float *m_dataFront;
    float *m_dataBack;

    float *m_dDataPtr;

    float *m_readPtr;
    float *m_writePtr;
    size_t pitch;

    cudaArray* m_cuWritePtr;

    size_t m_dataSize;

    dim3 m_dimBlock;
    dim3 m_dimGrid;

    FieldType m_type;

    virtual void flushUpdate();
};



/** Temporary fields for CudaBuffer
 *
 * One optimization of the implementation is to use textures and surfaces for
 * temporary fields, too. They are wrapped in this class
 */
class TmpFields{

public:

    /** this pointer is used to transfer the current fluid simulation from the
     * graphics card to the cpu*/
	float * m_displayPtr;
    /** size of array #m_displayPtr is pointing to */
	int m_displayDataSize;

    /** creates the temporary fields
     * @param width of the field
     * @param height of the field
     */
	TmpFields(int width, int height){ init(width, height);};
	virtual ~TmpFields(){free();};
    /**
     * flips front and back buffer
     */
	void flip();

    /** binds textures and surfaces to their cuda arrays */
	void bind();

private:
	cudaArray* tmp1SurfacePtr;
	cudaArray* tmp2SurfacePtr;
	cudaArray* divSurfacePtr;
	cudaChannelFormatDesc channelDescr;

	void init(int width, int height);
	void free();
};


#endif /* end of include guard: CUDABUFFER_H_5I18KWUB */

