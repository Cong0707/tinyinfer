// fp8 blockwise GEMM on tensor cores, behind a plain C ABI.
//
// Why a .dll and not a cubin: cutlass's 3.x collective API builds the kernel from
// templates AND needs host-side code to turn a problem shape into StrideA/StrideB and the
// two scale layouts (ScaleConfig::tile_atom_to_shape_SFA). A cubin carries only device
// code, so the host half has to live somewhere. This is the standard shape of the
// problem -- vLLM, SGLang and TensorRT-LLM all compile cutlass into a shared library for
// the same reason -- and it keeps zero torch dependency, unlike vLLM's _C.so.
//
// Scale granularity is 1 x 32 x 32: one scale per row of A, and one per 32x32 block of
// B, which is what the DeepSeek checkpoints store.
//
// Target is sm120 (GeForce Blackwell). On a laptop GPU the TMA path in cutlass's sm120
// kernels faults; on the datacenter parts this is the supported, exercised configuration.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;   // B is stored [N, K]
constexpr int AlignmentB = 16;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 8;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 32;
constexpr int ScaleGranularityK = 32;

using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// Tile K must equal the scale granularity K (a static_assert in
// sm120_blockwise_mma_builder.inl requires size<2>(TileShape) == ScaleGranularityK),
// and the checkpoints store one scale per 32 along K. So the K tile is 32.
// 128x128x32 needs 102400 B of shared memory, but an sm120 consumer part reports
// sharedMemPerBlockOptin = 101376 B (cutlass hard-codes the same constant in
// sm120_smem_capacity_bytes). The 1 KB gap makes the launch return kInvalid while
// cudaGetLastError() stays clean, which looks like a TMA fault and is not one.
// 128x64x32 fits comfortably and still tiles well at these N values.
using MmaTileShape = Shape<_128, _64, _32>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// A   [M, K]        e4m3
// SFA [M, K/32]     e8m0 (one row of scales per row of A)
// B   [N, K]        e4m3
// SFB [N/32, K/32]  e8m0
// D   [M, N]        bf16
//
// Returns 0 on success, a cutlass status otherwise, -100 on allocation failure.
//
// dllexport is not optional: Windows DLLs export nothing by default, so without it
// ctypes.CDLL loads the file and then fails on .tinyinfer_fp8_blockwise_gemm.
#if defined(_WIN32)
#define TINYINFER_EXPORT __declspec(dllexport)
#else
#define TINYINFER_EXPORT __attribute__((visibility("default")))
#endif

static char g_last_error[512] = {0};

extern "C" TINYINFER_EXPORT const char* tinyinfer_fp8_last_error() {
  return g_last_error;
}

extern "C" TINYINFER_EXPORT int tinyinfer_fp8_blockwise_gemm(
    const void* A, const void* SFA,
    const void* B, const void* SFB,
    void* D,
    int M, int N, int K,
    void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || (N % 32) != 0) return -2;

  auto problem_shape = cute::make_shape(M, N, K, 1);

  typename Gemm::GemmKernel::StrideA stride_A;
  typename Gemm::GemmKernel::StrideB stride_B;
  typename Gemm::GemmKernel::StrideC stride_C;
  typename Gemm::GemmKernel::StrideD stride_D;
  stride_A = cutlass::make_cute_packed_stride(stride_A, cute::make_shape(M, K, 1));
  stride_B = cutlass::make_cute_packed_stride(stride_B, cute::make_shape(N, K, 1));
  stride_C = cutlass::make_cute_packed_stride(stride_C, cute::make_shape(M, N, 1));
  stride_D = cutlass::make_cute_packed_stride(stride_D, cute::make_shape(M, N, 1));

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<const ElementA*>(A), stride_A,
       static_cast<const ElementB*>(B), stride_B,
       static_cast<const float*>(SFA), layout_SFA,
       static_cast<const float*>(SFB), layout_SFB},
      {{}, nullptr, stride_C, static_cast<ElementD*>(D), stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  cutlass::Status st = gemm.can_implement(arguments);
  if (st != cutlass::Status::kSuccess) {
    // Report which constraint failed instead of just the enum. cutlass checks
    // alignment, tile divisibility, cluster shape and the epilogue shape separately and
    // all of them return kInvalid, so the enum alone cannot be acted on.
    char msg[512];
    snprintf(msg, sizeof(msg),
             "can_implement=%d M=%d N=%d K=%d "
             "alignA=%d alignB=%d alignC=%d "
             "tileM=%d tileN=%d tileK=%d "
             "M%%tileM=%d N%%tileN=%d K%%tileK=%d "
             "cluster=%d%d%d",
             (int)st, M, N, K,
             (int)AlignmentA, (int)AlignmentB, (int)AlignmentC,
             (int)size<0>(MmaTileShape{}), (int)size<1>(MmaTileShape{}),
             (int)size<2>(MmaTileShape{}),
             M % (int)size<0>(MmaTileShape{}), N % (int)size<1>(MmaTileShape{}),
             K % (int)size<2>(MmaTileShape{}),
             (int)size<0>(ClusterShape{}), (int)size<1>(ClusterShape{}),
             (int)size<2>(ClusterShape{}));
    // Write into the caller's buffer: stderr from a DLL invoked through ctypes does not
    // reach the Python process reliably on Windows.
    if (g_last_error[0] == '\0') {
      for (int i = 0; i < 511 && msg[i]; ++i) g_last_error[i] = msg[i];
      g_last_error[511] = '\0';
    }
    return static_cast<int>(st);
  }

  size_t ws_bytes = Gemm::get_workspace_size(arguments);
  void* ws = nullptr;
  if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) return -100;

  cudaError_t e0 = cudaGetLastError();   // clear anything can_implement left behind
  (void)e0;
  st = gemm(arguments, ws, stream);
  cudaError_t e1 = cudaGetLastError();
  if (ws) cudaFree(ws);
  if (st != cutlass::Status::kSuccess) {
    snprintf(g_last_error, sizeof(g_last_error),
             "run=%d cuda=%s M=%d N=%d K=%d ws=%zu",
             (int)st, cudaGetErrorString(e1), M, N, K, ws_bytes);
  }
  return static_cast<int>(st);
}

extern "C" TINYINFER_EXPORT int tinyinfer_fp8_blockwise_gemm_workspace(void* A, void* SFA, void* B,
                                                     void* SFB, void* D, int M, int N,
                                                     int K) {
  (void)A; (void)SFA; (void)B; (void)SFB; (void)D;
  auto problem_shape = cute::make_shape(M, N, K, 1);
  typename Gemm::GemmKernel::StrideA stride_A;
  typename Gemm::GemmKernel::StrideB stride_B;
  typename Gemm::GemmKernel::StrideC stride_C;
  typename Gemm::GemmKernel::StrideD stride_D;
  stride_A = cutlass::make_cute_packed_stride(stride_A, cute::make_shape(M, K, 1));
  stride_B = cutlass::make_cute_packed_stride(stride_B, cute::make_shape(N, K, 1));
  stride_C = cutlass::make_cute_packed_stride(stride_C, cute::make_shape(M, N, 1));
  stride_D = cutlass::make_cute_packed_stride(stride_D, cute::make_shape(M, N, 1));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<const ElementA*>(A), stride_A,
       static_cast<const ElementB*>(B), stride_B,
       static_cast<const float*>(SFA), layout_SFA,
       static_cast<const float*>(SFB), layout_SFB},
      {{}, nullptr, stride_C, static_cast<ElementD*>(D), stride_D}};
  return static_cast<int>(Gemm::get_workspace_size(arguments));
}
