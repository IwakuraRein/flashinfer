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

#pragma once

#include <cstdint>

#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"

namespace moe::dev::routing {
namespace routingCustom {

void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchClusterKernel(Data const& data, void* stream);
void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist,
                                 void* stream);
void launchBlockScoresKernel(Data const& data, void* stream);
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream);
void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream);
void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream);

}  // namespace routingCustom
}  // namespace moe::dev::routing
