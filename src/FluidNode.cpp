#include "FluidNode.h"

#include <wrapper/raw_constructor.hpp>
#include <wrapper/WrapHelper.h>
#include <player/OGLSurface.h>
#include <base/ObjectCounter.h>
#include <player/Player.h>
#include <graphics/Pixel8.h>


using namespace boost::python;

char fluidNodeName[] = "FluidNode";


namespace avg
{

FluidNode::FluidNode(const ArgList& Args):
    RasterNode(),
    m_isInitialized(false)
{
    Args.setMembers(this);
    ObjectCounter::get()->incRef(&typeid(*this));
    Player::get()->registerPlaybackEndListener(this);
}

FluidNode::~FluidNode(){
    stop();
}

void FluidNode::stop(){
    if (m_pcudaDevice){
        delete m_pcudaDevice;
    }
    if (m_pCUDAPBO){
        delete m_pCUDAPBO;
    }
    if (m_PBO){
        delete m_PBO;
    }
    if (Player::exists()) {
        Player::get()->unregisterPlaybackEndListener(this);
    }
    ObjectCounter::get()->decRef(&typeid(*this));
}

void FluidNode::onPlaybackEnd()
{
    std::cout << "Playback End\n";
    stop();
}

void FluidNode::init(){
    m_pcudaDevice = new CudaDevice();
    IntPoint size(getSize().x, getSize().y);
    m_pTex = GLTexturePtr(new GLTexture(size, I8, false));
    m_pTex->enableStreaming();
    getSurface()->create(I8, m_pTex);
    newSurface();
    m_PBO = new PBO(size, I8, GL_DYNAMIC_DRAW);
    m_PBO->activate();
    m_pCUDAPBO = new TestCUDAPBO();
    m_pCUDAPBO->setPBO(m_PBO->getID());
}

void FluidNode::preRender(const VertexArrayPtr& pVA, bool bIsParentActive,
        float parentEffectiveOpacity){
    Node::preRender(pVA, bIsParentActive, parentEffectiveOpacity);
    if(m_isInitialized){
        m_pCUDAPBO->step();
        m_PBO->moveToTexture(*m_pTex);
        renderFX(getSize(), Pixel32(255, 255, 255, 255), false, false);
        calcVertexArray(pVA);
    }
}

void FluidNode::render(){
    if(!m_isInitialized){
        Player *player = Player::get();
        if(player->getFrameTime() > 50){
        m_isInitialized = true;
        init();
        }
    }else{
        Pixel8 color(255);
        blt32(getTransform(), getSize(), getEffectiveOpacity(),
                getBlendMode(), false);
    }
}

NodeDefinition FluidNode::createNodeDefinition(){
    return NodeDefinition("FluidNode", "rasternode", Node::buildNode<FluidNode>)
        ;
}

BOOST_PYTHON_MODULE(fluidnode)
{
    class_<FluidNode, bases<RasterNode>, boost::noncopyable>("FluidNode", no_init)
        .def("__init__", raw_constructor(createNode<fluidNodeName>))
        ;
}

AVG_PLUGIN_API void registerPlugin()
{
    initfluidnode(); // created by BOOST_PYTHON_MODULE
    object mainModule(handle<>(borrowed(PyImport_AddModule("__main__"))));
    object fluidModule(handle<>(PyImport_ImportModule("fluidnode")));
    mainModule.attr("fluidnode") = fluidModule;

    avg::NodeDefinition nodeDefinition = avg::FluidNode::createNodeDefinition();
    const char* allowedParentNodeNames[] = {"avg", "div", 0};
    avg::Player::get()->registerNodeType(nodeDefinition, allowedParentNodeNames);
}

} /* avg */
