#include <metal_stdlib>
using namespace metal;

// Boxes are (size, row offset, column offset). No full pixel-index buffer.
inline uint source_index(uint index, device const int* boxes, uint width) {
    uint col = index % 224;
    uint row = (index / 224) % 224;
    uint channel = (index / (224 * 224)) % 3;
    uint crop = index / (3 * 224 * 224);
    float scale = float(boxes[crop * 3]) / 224.0f;
    uint y = uint(float(row) * scale) + boxes[crop * 3 + 1];
    uint x = uint(float(col) * scale) + boxes[crop * 3 + 2];
    return (channel * width + y) * width + x;
}

kernel void cutouts_forward(device const float* image,
                            device const int* boxes,
                            device float* out,
                            constant long& width,
                            uint index [[thread_position_in_grid]]) {
    out[index] = image[source_index(index, boxes, uint(width))];
}

kernel void cutouts_backward(device const float* grad,
                             device const int* boxes,
                             device atomic_uint* result,
                             constant long& width,
                             uint index [[thread_position_in_grid]]) {
    uint destination = source_index(index, boxes, uint(width));
    // fp32 atomic add via integer CAS also works on earlier Apple GPUs.
    uint old = atomic_load_explicit(result + destination, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        result + destination, &old,
        as_type<uint>(as_type<float>(old) + grad[index]),
        memory_order_relaxed, memory_order_relaxed)) {}
}
