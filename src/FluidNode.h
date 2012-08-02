#ifndef FLUIDNODE_H_IMABZDHM
#define FLUIDNODE_H_IMABZDHM

#include <api.h>

#include <player/RasterNode.h>

namespace avg
{
class FluidNode: public RasterNode
{
public:
    FluidNode(const ArgList& Args);
    virtual ~FluidNode ();

    static NodeDefinition createNodeDefinition();

private:
    /* data */
};

} /* avg */


#endif /* end of include guard: FLUIDNODE_H_IMABZDHM */
