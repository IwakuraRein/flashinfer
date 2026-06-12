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

// Custom routing entry point. Kernel definitions and launch wrappers are split
// across trtllm_fused_moe_routing_custom_*.cu to reduce rebuild latency.

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
// Entry point
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                 data.mPtrTopKIds != nullptr)
      << "Routing kernel requires at least one input parameter";

  // When topK is already computed (mPtrTopKIds or mPtrTopKPacked without scores),
  // delegate to the shared post-topK pipeline which handles all path selection
  // (single-block, single-cluster, coop, multi-kernel) automatically.
  // No routing-method-specific logic needed.
  if (data.mPtrTopKIds != nullptr ||
      (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr)) {
    if (data.mPtrTopKIds != nullptr) {
      TVM_FFI_ICHECK(data.mPtrTopKWeights != nullptr)
          << "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for "
             "custom routing.";
    }
    runPostTopKPipeline(data, stream);
    return;
  }

  // After this point, input is mPtrScores (raw logits that need topK computation).
  TVM_FFI_ICHECK(data.mPtrScores != nullptr) << "Expected mPtrScores to be non-null at this "
                                                "point.";
  TVM_FFI_ICHECK(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
                 data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr)
      << "Custom routing kernel expects permuted idx and grouped Gemm launch config buffers";
  TVM_FFI_ICHECK_LE(data.mTopK, static_cast<int32_t>(MaxSupportedTopExperts))
      << "Routing kernel expects topK experts <= " << MaxSupportedTopExperts << ", got "
      << data.mTopK;
  TVM_FFI_ICHECK_LE(data.mNumExperts, static_cast<int32_t>(MaxSupportedExperts))
      << "Routing kernel expects #experts " << data.mNumExperts << " to be no more than "
      << MaxSupportedExperts << ".";

  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const useStaticBlock = data.mNumTokens <= BlockKernelMaxNumTokens;
  // Gate on the dispatched tier size, not the raw expert count.
  // Example: a model with 512 experts and topK=22 skips Tier<512,8> (topK too
  // large) and falls through to Tier<1024,32>.  queryDispatchedMaxExperts()
  // returns 1024 while mNumExperts is only 512.  The dynblock kernel sizes
  // shared memory proportional to maxExperts, so using the raw count (512 <=
  // 512, passes) would let it enter a 1024-expert specialization that exceeds
  // the smem budget.
  int32_t const dispatchedMaxExperts = queryDispatchedMaxExperts(data);
  bool const useDynBlock = !useStaticBlock && data.mNumTokens <= DynBlockKernelMaxNumTokens &&
                           dispatchedMaxExperts <= DynBlockKernelMaxNumExperts;
  bool const useSingleBlock = useStaticBlock || useDynBlock;
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= MaxNumTokensSingleClusterScores);

  // Split-topK path: block-per-token scores kernel + permutation-only cluster
  // kernel, the two overlapping via PDL.  This replaces the fused single-
  // cluster kernel for configs where it underperforms:
  //
  //   - Register pressure / spills: the fused kernel has every cluster warp
  //     carry K-sized per-thread arrays across the cluster barrier.  For
  //     Nemotron Super V3 (K=22) on B200 this yields 1088 spill slots and
  //     caps occupancy to 1 block/SM.
  //   - Idle-warp barrier stalls: at BS < cluster capacity the cluster
  //     still launches 256 warps, but only ~BS of them carry a token; the
  //     rest wait at __cluster_barrier_wait, dominating the profile (~75%
  //     of stalls at BS=64 on Nemotron).
  //   - No PDL overlap: everything happens in one kernel.
  //
  // Split-path structure:
  //   * routingIndicesBlockScoresKernel writes mPtrTopKPacked
  //     (1 block / token, 256 threads, only warp 0 carries K-sized regs).
  //   * runPostTopKPipeline selects the cluster kernel with
  //     LoadExpertIdxFromGlobal=true → permutation only.
  // The two kernels overlap via cudaTriggerProgrammaticLaunchCompletion()
  // and cudaGridDependencySynchronize().
  //
  // Dispatch rule (derived from a policy × (E, K) × BS sweep, see
  // `bench_routing_sweep.py` + `bench_routing_analyze.py`):
  //
  //   useSplit = useSingleCluster && !useSingleBlock && numExperts >= 160
  //
  // Why this rule (measured on B200 across {Default, Renormalize,
  // DeepSeekV3_nGrp1, MiniMax2} policies, BS ∈ [17, 256]):
  //   - At `numExperts >= 160` split wins on average for every policy:
  //     1.17× (Renormalize 160/8) up to 3.62× (DeepSeekV3 1024/32).
  //   - At `numExperts = 128`, split's two-kernel overhead (launch + PDL
  //     handoff) approximately cancels the single-cluster savings.  Default
  //     128/8 is ~1.0×; DeepSeekV3 / MiniMax2 128/8 show ~1.11× but
  //     Renormalize 128/8 is ~1.06× — not worth the complexity.
  //   - At numTokens < 17 the block/dynblock kernels already handle small
  //     BS optimally; this branch only fires when the cluster path would.
  //
  // Env-var override for benchmarking, applies to *both* the single-cluster
  // split path and the large-BS topK kernel choice:
  //   `FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN`
  //     unset / "auto" / "" → use the heuristic above (default)
  //     "1" / "on"          → force the block-per-token topK kernel wherever
  //                           applicable (ignores the numExperts guard, but
  //                           still requires the policy to opt in via
  //                           PolicyPairSupportsBlockPerToken)
  //     "0" / "off"         → force the fused single-cluster kernel for small
  //                           BS, and HistogramScoresKernel (warp-per-token)
  //                           for large BS
  //   Read once per process via a static local; invalid values silently fall
  //   back to "auto".
  enum class ForceMode { kAuto, kOn, kOff };
  static ForceMode const forceMode = [] {
    char const* raw = std::getenv("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN");
    if (raw == nullptr) return ForceMode::kAuto;
    std::string v = raw;
    if (v == "1" || v == "on" || v == "ON") return ForceMode::kOn;
    if (v == "0" || v == "off" || v == "OFF") return ForceMode::kOff;
    return ForceMode::kAuto;
  }();

  // Does the currently-active policy pair implement the block-per-token
  // interface?  Queried via PolicyPairSupportsBlockPerToken — policies that
  // don't opt in (no applyToSmem / applyWithAux specialisation) will force
  // this branch to false and fall back to the fused single-cluster kernel.
  bool const policySupportsBlockPerToken = queryPolicySupportsBlockPerToken(data);
  if (forceMode == ForceMode::kOn && !policySupportsBlockPerToken) {
    FLASHINFER_WARN(
        "FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN is set but the active routing policy does not "
        "support block-per-token; the request is ignored.");
  }

  bool useSplitTopKPath = useSingleCluster && !useSingleBlock && policySupportsBlockPerToken &&
                          (data.mNumExperts >= NumExperts160Experts);
  if (forceMode == ForceMode::kOn && useSingleCluster && !useSingleBlock &&
      policySupportsBlockPerToken) {
    useSplitTopKPath = true;
  } else if (forceMode == ForceMode::kOff) {
    useSplitTopKPath = false;
  }
  if (!useSingleCluster && !useSingleBlock) {
    TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr)
        << "When #tokens is large, `mPtrTopKPacked` is a required input.";
    TVM_FFI_ICHECK(data.mPtrExpertCounts != nullptr)
        << "When #tokens is large, `mPtrExpertCounts` is a required input.";
  } else if (useSplitTopKPath) {
    // BlockScoresKernel unconditionally writes topK results to mPtrTopKPacked
    // (the downstream runPostTopKPipeline then reads it as pre-computed topK).
    // Unlike the large-BS path above, this branch is gated on useSingleCluster,
    // so we need a separate check here.  mPtrExpertCounts has its own
    // `nullptr` guard inside the kernel (histogram reset), so it remains
    // optional and doesn't need a check.
    TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr)
        << "The block-per-token split path requires `mPtrTopKPacked` to be non-null "
           "(BlockScoresKernel writes the topK result into it).";
  }

  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));

  // We need a mutable copy since `data` is const.
  Data mutableData = data;

  if (useSplitTopKPath) {
    // Step 1: scores → mPtrTopKPacked via a block-per-token kernel that
    // mirrors TRT-LLM's deepseek_v3_topk_kernel (no-groups path) layout:
    //   - One block per token, kBlockScoresKernelBlockDim threads per block.
    //   - Block-parallel preprocess (via PreprocessPolicy::applyToSmem).
    //   - Warp 0 alone does the sort-based reduceTopK<K, N=ceil(E/32)> plus
    //     the postprocess (via PostprocessPolicy::applyWithAux).
    //   - Only warp 0 holds the K-sized register arrays, so per-block
    //     register pressure stays low and occupancy high.
    //
    // Why 256 threads (kBlockScoresKernelBlockDim): larger blocks (e.g. 512)
    // would reserve the ~99-register budget across more threads and limit
    // occupancy to 1 block/SM.  256 threads gives 2 blocks/SM with the same
    // per-warp workload, doubling arithmetic throughput during the preprocess
    // phase.  For E=512 the grid-stride loop iterates 2× per thread; for
    // E<=256 it iterates once.
    //
    // The kernel is generic in (PreprocessPolicy, PostprocessPolicy) — any
    // registered policy pair in RoutingCustomPolicy.cuh works here.
    launchBlockScoresKernel(mutableData, stream);

    // Step 2: delegate the permutation pipeline to runPostTopKPipeline, which
    //   will pick the cluster kernel (LoadExpertIdxFromGlobal=true branch) for
    //   small numTokens.  The cluster kernel in that branch reads topK from
    //   mPtrTopKPacked instead of doing topK itself, so it does not carry the
    //   K-sized register arrays or suffer from idle-warp barrier waits.
    //
    //   Clear mPtrScores so runPostTopKPipeline takes the pre-computed-topK
    //   path (we've already written mPtrTopKPacked above).
    mutableData.mPtrScores = nullptr;
    runPostTopKPipeline(mutableData, stream);
    return;
  }

  if (useDynBlock) {
    launchDynBlockKernel(mutableData, numThreadsHist, stream);
  } else if (useStaticBlock) {
    launchBlockKernel(mutableData, numThreadsHist, stream);
  } else if (useSingleCluster) {
    launchClusterKernel(mutableData, stream);
  } else {
    uint32_t const maxNumBlocks = 1024;

    // TopK kernel selection for the large-BS path.
    //
    // The default is `routingIndicesHistogramScoresKernel` (warp-per-token,
    // grid-stride over numTokens): well-suited to the large-BS regime where
    // there's naturally one warp's worth of work per token.
    //
    // For policies that support the block-per-token interface and have E >= 256,
    // `routingIndicesBlockScoresKernel` (1 block/token, warp-0 sort-based topK)
    // can be faster because:
    //   - Only warp 0 carries the K-sized topK register arrays (saves ~40
    //     regs/thread for K=22, avoids spills even at large E).
    //   - The bitonic-sort reduceTopK over N=ceil(E/32) elements per lane
    //     beats the K sequential warp reductions a warp-per-token layout
    //     does for high-K configs like Nemotron.
    //
    // The trade-off is one block per token — at BS = 4096 that's 4096 blocks
    // vs ~128 blocks for warp-per-token.  Above BS ≈ 1024 block-per-token
    // starts oversubscribing SMs and loses to the warp-per-token layout.
    //
    // Dispatch rule (derived from the same sweep as the cluster path):
    //
    //   useBlockScores = (E >= NumExperts1024Experts)
    //                 || (E >= NumExperts256Experts && BS <= 1024)
    //
    // The E=1024/K=32 tier is a special corner: block-per-token wins at
    // every BS (1.95×–5.17×) because fused warp-per-token topK explodes on
    // register pressure when K=32.  For E ∈ [256, 576], speedups are
    // 1.01×–1.75× at BS <= 1024 but collapse to 0.77×–1.03× at BS >= 2048,
    // hence the BS cap.
    bool useBlockScoresForTopK =
        policySupportsBlockPerToken &&
        ((data.mNumExperts >= NumExperts1024Experts) ||
         (data.mNumExperts >= NumExperts256Experts && data.mNumTokens <= 1024));
    if (forceMode == ForceMode::kOn && policySupportsBlockPerToken) {
      useBlockScoresForTopK = true;
    } else if (forceMode == ForceMode::kOff) {
      useBlockScoresForTopK = false;
    }
    if (useBlockScoresForTopK) {
      launchBlockScoresKernel(mutableData, stream);
    } else {
      launchHistogramScoresKernel(mutableData, maxNumBlocks, numThreadsHist, stream);
    }

    bool const canUseCoop =
        (smMajor >= 9) && (data.mNumExperts <= 1024) && (data.mPtrPermutedIdxSize != nullptr);
    bool useCoop = false;
    CoopLaunchSMCounts coopLaunchSMCounts{0, 0};
    int numBlocksCoop = 0;

    if (canUseCoop) {
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      coopLaunchSMCounts = getCoopLaunchSMCounts(smCount);
      numBlocksCoop = coopLaunchSMCounts.moeSms;
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      logCoopLaunchSMCounts(coopLaunchSMCounts);
      launchInitExpertCounts(mutableData, numThreadsHist, stream);
      launchCoopKernel(mutableData, numBlocksCoop, numThreadsHist, stream);
    } else {
      uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      launchHistogramKernel(mutableData, numBlocksHistogram, numThreadsHist, stream);
      launchOffsetsKernel(mutableData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
