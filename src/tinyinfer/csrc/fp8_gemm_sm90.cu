// fp8 blockwise GEMM for Hopper (sm90), behind a plain C ABI.
//
// Follows cutlass's own 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling:
// the sm90 path needs the FP8Blockwise kernel schedule and the scale config derived from
// the tile shape, not a hand-written granularity. On sm90 the scale granularity is tied
// to the tile (128 along K here), which is why this and the sm120 file are separate
// kernels rather than one template.
//
// The sm120 counterpart lives in fp8_gemm.cpp; which one loads is decided in Python.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 8;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementBlockScale = float;
using ElementCompute = float;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;

using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(TileShape{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#if defined(_WIN32)
#define TINYINFER_EXPORT __declspec(dllexport)
#else
#define TINYINFER_EXPORT __attribute__((visibility("default")))
#endif

static char g_last_error[512] = {0};

extern "C" TINYINFER_EXPORT const char* tinyinfer_fp8_last_error() {
  return g_last_error;
}

// A [M,K] e4m3, SFA [M,K/128], B [N,K] e4m3, SFB [N/128,K/128], D [M,N] bf16.
extern "C" TINYINFER_EXPORT int tinyinfer_fp8_blockwise_gemm(
    const void* A, const void* SFA,
    const void* B, const void* SFB,
    void* D, int M, int N, int K, void* stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (M <= 0 || N <= 0 || K <= 0) return -2;

  auto problem_shape = cute::make_shape(M, N, K, 1);

  typename Gemm::GemmKernel::StrideA stride_A;
  typename Gemm::GemmKernel::StrideB stride_B;
  typename Gemm::GemmKernel::StrideC stride_C;
  typename Gemm::GemmKernel::StrideD stride_D;
  // cutlass 4.x dropped util/packed_stride.hpp. StrideA/B/C/D here are STATIC types
  // (dynamic strides would be Stride<int,...>); they carry no runtime values, and the
  // kernel derives the offsets from the problem shape. Default construction is correct.
  stride_A = typename Gemm::GemmKernel::StrideA{};
  stride_B = typename Gemm::GemmKernel::StrideB{};
  stride_C = typename Gemm::GemmKernel::StrideC{};
  stride_D = typename Gemm::GemmKernel::StrideD{};

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {static_cast<const ElementA*>(A), stride_A,
       static_cast<const ElementB*>(B), stride_B,
       static_cast<const ElementBlockScale*>(SFA), layout_SFA,
       static_cast<const ElementBlockScale*>(SFB), layout_SFB},
      {{}, nullptr, stride_C, static_cast<ElementD*>(D), stride_D}};
  arguments.epilogue.thread.alpha_ptr = nullptr;

  Gemm gemm;
  cutlass::Status st = gemm.can_implement(arguments);
  if (st != cutlass::Status::kSuccess) {
    snprintf(g_last_error, sizeof(g_last_error),
             "can_implement=%d M=%d N=%d K=%d", (int)st, M, N, K);
    return static_cast<int>(st);
  }
  size_t ws_bytes = Gemm::get_workspace_size(arguments);
  void* ws = nullptr;
  if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) return -100;
  st = gemm(arguments, ws, stream);
  if (ws) cudaFree(ws);
  if (st != cutlass::Status::kSuccess) {
    snprintf(g_last_error, sizeof(g_last_error), "run=%d M=%d N=%d K=%d",
             (int)st, M, N, K);
  }
  return static_cast<int>(st);
}
