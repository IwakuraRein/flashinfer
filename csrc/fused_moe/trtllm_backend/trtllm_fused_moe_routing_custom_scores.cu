/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <string>

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "trtllm_fused_moe_routing_custom.h"
#include "tvm_ffi_utils.h"

namespace moe::dev::routing {
namespace routingCustom {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3. HistogramScores kernel — computes TopK from raw scores and initializes expert counts.
//    Used as step 1 of the multi-kernel pipeline when input is raw logits.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  // Cap actual thread count at 1024 when MaxNumExperts > 1024.
  static constexpr int NumThreadsBlock =
      KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;

  // VecSize stays based on MaxNumExperts — each warp still processes all experts for one token.
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  // Use NumThreadsBlock (actual thread count) for grid-stride warp/thread addressing
  int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
  int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait on primary grid.
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

  // initialize the mPtrExpertCounts — use NumThreadsBlock for grid-stride
  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreadsBlock;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

  // in this case, each warp represents a token, and we use a grid-stride loop
  // over all warps/tokens
  BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
  int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                     KernelParams::MaxNumTopExperts>(
        warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
        params.mPtrScores + scoreOffset, params);

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(warpTopKScore[laneIdx]),
                                          static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
      // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
      // -- same indexing as mPtrTopKPacked.
      if (params.mPtrRoutingReplayOut != nullptr) {
        params.mPtrRoutingReplayOut[tokenIdx * params.mTopK + laneIdx] =
            static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger secondary kernel AFTER writing all packed scores, so the next kernel
  // (routingIndicesHistogramKernel) sees the completed mPtrTopKPacked writes.
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist,
                                 void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                        numThreadsHist,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3b. BlockScores kernel — one block per token, block-parallel preprocess + warp-0 sort-based topK.
//
// Motivation: for small numTokens with high topK (e.g. Nemotron Super V3:
// BS<=256, E=512, K=22), the per-warp-per-token `routingIndicesHistogramScoresKernel`
// suffers from (a) high register pressure — *every* warp in a block carries
// the K-sized topK arrays (~99 regs/thread → 25% occupancy), and (b) low
// arithmetic intensity per warp during preprocess (each lane redundantly
// processes VecSize=MaxNumExperts/32 experts into registers).
//
// This kernel instead mirrors the design used by the no-groups path of
// TRT-LLM's `deepseek_v3_topk_kernel`:
//   Phase 1: parallelise the preprocess across all threads via a grid-stride
//            loop (PreprocessPolicy::applyToSmem), writing per-expert topK
//            key (and optional aux data) into smem.
//   Phase 2: warp 0 alone does the topK via a sort-based reduceTopK with
//            N = ceil(MaxNumExperts / WarpSize) elements per lane.
//   Phase 3: warp 0 applies the postprocess (PostprocessPolicy::applyWithAux),
//            in-place modifying topScores.
// Only warp 0 carries the K-sized register arrays, so per-block register
// pressure stays low (~64 regs/thread) and occupancy rises to 2 blocks/SM.
//
// The kernel is generic in (PreprocessPolicy, PostprocessPolicy).  The
// auxiliary smem array is only allocated when the policy pair requires it
// (today: only ScaledSumNormalizePostprocess — see PolicyPairNeedsAux trait
// in RoutingCustomPolicy.cuh).  Otherwise, the "aux" pointer aliases the
// "biased" pointer and no extra smem is reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Block dimension for routingIndicesBlockScoresKernel.  Chosen so that:
//   - __launch_bounds__ can reserve the per-thread register budget
//     (each SM then fits 2 blocks at 64 regs/thread, the measured sweet spot).
//   - SoftmaxPreprocess::applyToSmem<kBlockDim> can size its cub::BlockReduce
//     temp storage at compile time.
// The value is kept as a compile-time constant (rather than a runtime
// parameter) so the kernel and its launcher share a single source of truth;
// the internal Softmax path would not function correctly if the actual
// blockDim.x disagreed with this value.
constexpr int kBlockScoresKernelBlockDim = 256;

template <typename KernelParams>
__global__ void __launch_bounds__(kBlockScoresKernelBlockDim)
    routingIndicesBlockScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using ExpertSelect = typename KernelParams::ExpertSelectPolicy;
  using PreProc = typename ExpertSelect::PreprocessPolicy;
  using PostProc = typename ExpertSelect::PostprocessPolicy;
  using BaseType = typename ExpertSelect::template BaseType<InputT>;

  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int MaxNumTopExperts = KernelParams::MaxNumTopExperts;
  // One chunk per warp lane — each lane of warp 0 holds NumChunks experts.
  // For E=512 this is 16; for E=1024 it's 32; for E=2048 it's 64 — which hits
  // Sort<N>'s 64-element cap (the static_assert inside topk::reduceTopK).
  static constexpr int NumChunks = (MaxNumExperts + WarpSize - 1) / WarpSize;

  // Compile-time opt-out: the macro dispatch instantiates this kernel across
  // every (PreProc, PostProc, tier) combination, but we only emit a real
  // kernel body for policy pairs that implement the block-per-token interface
  // (PolicyPairSupportsBlockPerToken<PreProc, PostProc>::value).  For the
  // other instantiations we emit an empty kernel body.  These instantiations
  // are never reached at runtime because `run()` gates the host-side launch
  // on the same trait check, but compiling them as no-ops keeps this kernel
  // generic across the whole policy matrix.
  //
  // The expert count is bounded by Sort<N>'s N ≤ 64 cap, i.e. MaxNumExperts
  // ≤ 64 * WarpSize = 2048.  This matches `MaxSupportedExperts` in
  // RoutingCustomPolicy.cuh, so every tier declared in `PolicyTraits` fits
  // and no extra guard is needed here.
  static constexpr bool kSupported = PolicyPairSupportsBlockPerToken<PreProc, PostProc>::value;
  static_assert(NumChunks <= 64,
                "routingIndicesBlockScoresKernel: MaxNumExperts must be <= 64 * WarpSize = 2048 "
                "(Sort<N>'s upper bound inside topk::reduceTopK)");
  if constexpr (!kSupported) {
    return;
  } else {
    // Allocate smemAux only when the postprocess actually reads it.  For every
    // other (pre, post) combination we save MaxNumExperts × 4 B of smem by
    // letting `auxPtr` alias `smemBiased`.
    static constexpr bool kNeedsAux = PolicyPairNeedsAux<PreProc, PostProc>::value;
    static constexpr int kAuxSize = kNeedsAux ? MaxNumExperts : 1;

    static constexpr float invalidScoreFloat = -INFINITY;

    // Per-expert smem arrays:
    //   smemBiased[e] = topK selection key for expert e
    //   smemAux[e]    = auxiliary data for expert e (only used / written when
    //                   PolicyPairNeedsAux<PreProc, PostProc>::value is true)
    __shared__ BaseType __attribute((aligned(128))) smemBiased[MaxNumExperts];
    __shared__ BaseType __attribute((aligned(128))) smemAuxStorage[kAuxSize];
    BaseType* auxPtr = kNeedsAux ? smemAuxStorage : smemBiased;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    int32_t const laneIdx = cutlass::arch::LaneId();
    // blockDim.x is always kBlockScoresKernelBlockDim (a multiple of WarpSize),
    // so `threadIdx.x / WarpSize` is already uniform within a warp — no
    // `__shfl_sync` broadcast needed.
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const tokenIdx = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (params.mUsePdl) {
      cudaGridDependencySynchronize();
    }
#endif

    // Reset `mPtrExpertCounts` (histogram needed by the downstream coop /
    // multi-kernel permutation paths).  Block 0 does this across its threads;
    // other blocks skip to avoid redundant writes.  Size is 2*numExperts (the
    // array is used both for the final histogram and the tile-offset histogram).
    if (blockIdx.x == 0 && params.mPtrExpertCounts != nullptr) {
      int32_t const expertCountsNum = 2 * params.mNumExperts;
      for (int i = threadIdx.x; i < expertCountsNum; i += blockDim.x) {
        params.mPtrExpertCounts[i] = 0;
      }
    }

    // Phase 1: block-parallel preprocess.  Dispatches on PreProc so the kernel
    // works for all registered preprocess policies (single-pass NoOp / Sigmoid /
    // SigmoidBias, two-pass Softmax).
    int64_t const scoreBase = int64_t{tokenIdx} * int64_t{params.mNumExperts};
    if constexpr (std::is_same_v<PreProc, SoftmaxPreprocess>) {
      // Softmax needs block-level reductions; pass the block size as a template
      // parameter so it can construct cub::BlockReduce with the right size.
      // We rely on kBlockScoresKernelBlockDim matching the actual launch
      // blockDim.x — enforced by the launcher below.
      PreProc::template applyToSmem<kBlockScoresKernelBlockDim>(
          block, params.mPtrScores + scoreBase, params.mNumExperts, smemBiased, auxPtr,
          params.mExpertSelectParams.mPreprocessParams);
    } else {
      PreProc::applyToSmem(block, params.mPtrScores + scoreBase, params.mNumExperts, smemBiased,
                           auxPtr, params.mExpertSelectParams.mPreprocessParams);
    }

    __syncthreads();

    // Phase 2: warp-0 sort-based topK over NumChunks elements per lane.
    // Warps 1..N-1 are done and can exit; only warp 0 carries the K-sized
    // register arrays from here on.
    if (warpIdx != 0) {
      return;
    }

    BaseType localScores[NumChunks];
    int32_t localIdx[NumChunks];
#pragma unroll
    for (int ii = 0; ii < NumChunks; ++ii) {
      int const eIdx = ii * WarpSize + laneIdx;
      localIdx[ii] = eIdx;
      localScores[ii] = eIdx < params.mNumExperts ? smemBiased[eIdx] : BaseType{invalidScoreFloat};
    }

    BaseType topScores[MaxNumTopExperts];
    int32_t topExperts[MaxNumTopExperts];
    topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
                     /*minValue=*/BaseType{invalidScoreFloat}, params.mTopK);

    // Phase 3: postprocess.  Reads the per-expert aux data (if the policy needs
    // it) and in-place modifies topScores.
    PostProc::applyWithAux(warp, topScores, topExperts, laneIdx, params.mTopK, auxPtr,
                           params.mExpertSelectParams.mPostprocessParams);

    // Phase 4: write packed (score, expertIdx) for the downstream permutation stage.
    if (laneIdx < params.mTopK) {
      int32_t const expertIdx = topExperts[laneIdx];
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(topScores[laneIdx]),
                                          static_cast<int16_t>(expertIdx)};
      params.mPtrTopKPacked[int64_t{tokenIdx} * int64_t{params.mTopK} + laneIdx] = packedScore;
      // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
      // -- same indexing as mPtrTopKPacked.
      if (params.mPtrRoutingReplayOut != nullptr) {
        params.mPtrRoutingReplayOut[int64_t{tokenIdx} * int64_t{params.mTopK} + laneIdx] =
            static_cast<int16_t>(expertIdx);
      }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (params.mUsePdl) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
  }  // end `if constexpr (kSupported)` else branch
}

void launchBlockScoresKernel(Data const& data, void* stream) {
  // Custom dispatch that does NOT clamp blockDim to the dispatched tier's
  // expert count (unlike `LAUNCH_ROUTING_CUSTOM`).  The kernel's layout —
  // kBlockScoresKernelBlockDim threads with a grid-stride loop over experts —
  // is decoupled from MaxNumExperts; oversizing blockDim would waste
  // registers and shrink occupancy.  The blockDim is intentionally fixed to
  // the kernel-side constant so host and device agree (SoftmaxPreprocess
  // sizes its cub::BlockReduce with the same value at compile time).
  dispatchRoutingPolicy(data, [&](auto preProc_, auto postProc_, char const* policyName_) {
    using PreProc_ = decltype(preProc_);
    using PostProc_ = decltype(postProc_);
    using Pairs_ = typename PolicyTraits<PreProc_, PostProc_>::Pairs;
    bool dispatched_ =
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {
          LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/false, routingIndicesBlockScoresKernel,
                                       /*gridDim=*/data.mNumTokens,
                                       /*blockDim=*/kBlockScoresKernelBlockDim,
                                       /*smemSize=*/0, stream, PreProc_, PostProc_,
                                       decltype(eTag_)::value, decltype(kTag_)::value);
        });
    if (!dispatched_) {
      FLASHINFER_WARN(
          "No compiled tier covers numExperts=%d topK=%d for policy %s in "
          "launchBlockScoresKernel.",
          data.mNumExperts, data.mTopK, policyName_);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
