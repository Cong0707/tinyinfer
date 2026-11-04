
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using ArchTag = cutlass::arch::Sm120;
using MainloopOpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeMxf4Sm120;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape,
    Shape<_128, _64>, float, float, ElementC, cutlass::layout::RowMajor*, 8,
    ElementC, cutlass::layout::RowMajor*, 8, EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, MainloopOpClass, ElementA, cutlass::layout::RowMajor*, 32,
    ElementB, cutlass::layout::ColumnMajor*, 32, float,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Scheduler = typename Gemm::GemmKernel::TileScheduler;
using SchedParams = typename Scheduler::Params;

__global__ void probe() {}
int main() { return 0; }
