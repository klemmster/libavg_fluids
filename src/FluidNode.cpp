#include "FluidNode.h"

#include <wrapper/raw_constructor.hpp>
#include <wrapper/WrapHelper.h>

using namespace boost::python;

char fluidNodeName[] = "FluidNode";

namespace avg
{

FluidNode::FluidNode(const ArgList& Args):
    RasterNode()
{
    Args.setMembers(this);
}

FluidNode::~FluidNode(){

}

NodeDefinition FluidNode::createNodeDefinition(){
    return NodeDefinition("FluidNode", Node::buildNode<FluidNode>)
        .extendDefinition(RasterNode::createDefinition())
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
