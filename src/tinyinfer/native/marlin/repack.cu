// GPTQ-format to Marlin-format weight repack, ported from vLLM's
// quantization/marlin/gptq_marlin_repack.cu. It runs once per expert at load time
// (~11us each) to put the packed int4 weights into the interleaved layout the Marlin
// GEMV reads.
//
// repack_body.inc is the upstream kernel body verbatim (it owns its own namespace);
// this file adds the instantiation this engine needs and keeps torch out of the build.
#include "marlin.cuh"
#include "marlin_compat.h"
#include "repack_body.inc"

namespace marlin {

// <threads, num_bits, is_a_8bit> -- 4-bit weights, 16-bit activations.
template __global__ void gptq_marlin_repack_kernel<256, 4, false>(
    uint32_t const* __restrict__, uint32_t* __restrict__, int, int);

}  // namespace marlin
