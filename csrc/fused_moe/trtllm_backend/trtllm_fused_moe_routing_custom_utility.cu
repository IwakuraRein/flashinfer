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
// 4. Coop kernel — cooperative histogram + offsets via grid-sync.
//
// The coop kernel only performs the post-topK permutation pipeline (histogram, prefix-scan,
// index writes). It does NOT compute topK — it reads pre-computed results from mPtrTopKPacked
// or mPtrTopKIds. Therefore, only the NumExperts template tier matters (it sizes shared memory
// arrays and determines the thread count). The NumTopExperts tier is fixed at NumTop8Experts
// because the kernel uses a hardcoded MaxExpandedIdxPerThread=64 and processes all expanded
// indices (mNumTokens * mTopK) via a grid-stride loop with runtime bounds checking, regardless
// of the compile-time MaxNumTopExperts value.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream) {
  if (data.mNumExperts <= NumExperts128Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts128Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts160Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts160Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts256Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts256Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts384Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts384Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts512Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts512Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts576Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts576Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts1024Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts1024Experts, NumTop8Experts);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Coop kernel does not support numExperts > " << NumExperts1024Experts << ", got "
        << data.mNumExperts;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 5-7. Launch wrappers for shared kernels (defined in RoutingKernel.cuh):
//      - InitExpertCounts (zero expert counts)
//      - Histogram kernel (histogram from packed TopK)
//      - Offsets kernel (prefix-scan + permutation)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingInitExpertCounts,
                                  (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesHistogramKernel, numBlocksHistogram,
                                  numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesOffsetsKernel, numBlocksOffsets,
                                  numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
