// Marlin fp4 GEMV instances for this engine's shapes.
//
// The kernel body lives in marlin_template.h (ported from vLLM's quantization/marlin);
// kernel.h only declares the template. Both must be included for an explicit
// instantiation to have a definition -- with kernel.h alone the cubin comes out at 2KB
// and the symbol resolves to nothing.
//
// Only the specialisations this engine asks for are instantiated, which keeps the cubin
// small and the build to seconds instead of the minutes a full kernel_selector.h costs.
#include "kernel.h"
#include "marlin_template.h"

namespace marlin {

// template <a_type, b_type, c_type, s_type, threads, thread_m_blocks, thread_n_blocks,
//           thread_k_blocks, m_block_size_8, stages, group_blocks, is_zp_float>
//
// bf16 activation, fp4 weight (E2M1), bf16 output, e8m0 scales -- what this engine runs
// today (w4a16: the fp8 activation quantisation is skipped).
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                                vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                                256, 1, 8, 8, true, 4, 2, false>(
    MARLIN_KERNEL_PARAMS);

// fp8 (e4m3) activation, same everything else -- the w4a8 form the checkpoints actually
// define. This is the combination marlin.cu gates on SM89 or SM12x (elsewhere it says
// W4A8 is slower than W4A16 and refuses), which is why the a_type check below exists at
// all. Instantiating it costs one more template in the cubin and lets the fp8-quantised
// path keep its arithmetic instead of dropping to w4a16 for speed.
template __global__ void Marlin<vllm::kFE4M3fn.id(), vllm::kFE2M1f.id(),
                                vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                                256, 1, 8, 8, true, 4, 2, false>(
    MARLIN_KERNEL_PARAMS);

}  // namespace marlin
