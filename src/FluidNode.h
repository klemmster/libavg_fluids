#ifndef FLUIDNODE_H_IMABZDHM
#define FLUIDNODE_H_IMABZDHM

#define AVG_PLUGIN

#include <api.h>

#include <base/IPlaybackEndListener.h>

#include <player/RasterNode.h>
#include <graphics/PBO.h>
#include <graphics/GLTexture.h>
#include <graphics/Bitmap.h>

#include "CudaDevice.h"
#include "TestCUDAPBO.h"

namespace avg
{
class FluidNode: public RasterNode, IPlaybackEndListener
{
public:
    FluidNode(const ArgList& Args);
    virtual ~FluidNode ();

    virtual void preRender(const VertexArrayPtr& pVA, bool bIsParentActive,
            float parentEffectiveOpacity);
    virtual void render();
    virtual void onPlaybackEnd();
    static NodeDefinition createNodeDefinition();

protected:
    void init();
    void stop();

private:
    //Load and initialize the first cuda capable device
    CudaDevice *m_pcudaDevice;
    PBO *m_PBO;
    TestCUDAPBO *m_pCUDAPBO;
    GLTexturePtr m_pTex;
    bool m_isInitialized;
};

} /* avg */


#endif /* end of include guard: FLUIDNODE_H_IMABZDHM */
