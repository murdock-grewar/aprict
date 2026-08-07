
#include "base.h"
#include "geometry.h"
#include "FORBILD_volumes.h"

// This kernel is intended to be used with a grid of size (gridX, 1, 1) blocks
// and block size (blockX, 1, 1)
template<
    typename Ts   // Can be FORBILD_Shape, which is inefficient but allows passing a bunch of different shapes at once. Or can be a specific shape type, like FORBILD_Sphere, et cetera
>
__global__ void get_bboxes_FORBILD_shapes(
        const Ts*               shapes_glob,     // shapes array
        const unsigned int      num_shapes,      // number of shapes
        bbox*                   bbs
)
{
    const unsigned long tint = blockDim.x * blockIdx.x + threadIdx.x;
    if (tint >= num_shapes) { return; }
    Ts shap = shapes_glob[tint];
    bbs[tint] = get_bbox(shap);
}
