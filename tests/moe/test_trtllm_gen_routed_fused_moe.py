"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
from typing import Literal
import torch

from flashinfer import (
    RoutingMethodType,
    ActivationType,
    fp4_quantize,
    mxfp8_quantize,
    shuffle_matrix_a,
)
from flashinfer.fused_moe import (
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.utils import device_support_pdl

from .test_trtllm_gen_fused_moe import (
    routing_reference_renormalize,
    routing_reference_renormalize_naive,
    routing_reference_topk,
)

from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("intermediate_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("num_experts", [128, 256])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
        RoutingMethodType.TopK,
    ],
)
@pytest.mark.parametrize("quant_mode", ["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"])
def test_trtllm_gen_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    if quant_mode == "NvFP4xNvFP4":
        hidden_states, hidden_states_scale = fp4_quantize(
            hidden_states,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
    elif quant_mode == "MxFP4xMxFP8":
        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0
    else:  # MxFP4xBf16
        hidden_states_scale = None
        hidden_states_global_scale = 1.0

    w13 = (
        torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    if quant_mode == "NvFP4xNvFP4":
        w13, w13_scale = fp4_quantize(
            w13,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            torch.tensor([448.0 * 6.0], device=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0 / 448.0 / 6.0
        w2_global_scale = 1.0 / 448.0 / 6.0
    else:
        w13, w13_scale = fp4_quantize(
            w13, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
        )
        w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0
        w2_global_scale = 1.0

    output1_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output1_scale_gate_scalar = torch.tensor(
        [hidden_states_global_scale * w13_global_scale] * num_experts, device=device
    )
    output2_scale_scalar = torch.tensor(
        [hidden_states_global_scale * w2_global_scale] * num_experts, device=device
    )

    reference_output = trtllm_fp4_block_scale_moe(
        routing_logits,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2,
        w2_scale,
        None,  # w2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,  # act_type
        None,
    )[0].to(torch.float)

    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights.view(num_tokens, num_experts)[
        torch.arange(num_tokens).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    packed_tensor = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    output = trtllm_fp4_block_scale_routed_moe(
        packed_tensor,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2,
        w2_scale,
        None,  # w2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,  # act_type
        None,
    )[0].to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-3, atol=1e-3)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 6, f"Mismatch percentage is {mismatch_pct:.2f}"


@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
    ],
)
def test_trtllm_gen_fp8_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
):
    """Test FP8 block scale routed MoE matches standard routing."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    # Generate random routing logits for reference
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    # Generate random hidden states in FP8
    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    hidden_states = hidden_states_bf16.to(torch.float8_e4m3fn)

    # Generate block scales for hidden states: [hidden_size // 128, num_tokens]
    hidden_states_scale = torch.ones(
        hidden_size // 128, num_tokens, device=device, dtype=torch.float32
    )

    # Generate FP8 weights
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.float8_e4m3fn)

    # Generate block scales for weights
    gemm1_weights_scale = torch.ones(
        num_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights_scale = torch.ones(
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
        device=device,
        dtype=torch.float32,
    )

    # Run reference with routing_logits
    reference_output = trtllm_fp8_block_scale_moe(
        routing_logits,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        routing_method_type.value,
        False,  # use_shuffled_weight
        0,  # weight_layout
        enable_pdl,
    ).to(torch.float)

    # Compute routing using reference implementation
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights_ref = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights_ref = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights_ref = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_ref.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # Pack topk_ids and expert_weights into single tensor
    # Format: (expert_id << 16) | (weight_bf16.view(int16))
    packed_topk_ids = (topk_ids << 16) | expert_weights.view(torch.int16).to(
        torch.int32
    )

    # Run with pre-computed routing (packed format)
    output = trtllm_fp8_block_scale_routed_moe(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=False,
        weight_layout=0,
        enable_pdl=enable_pdl,
    ).to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-2, atol=1e-2)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"


@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize(
    "routing_method_type",
    [
        RoutingMethodType.Renormalize,
    ],
)
def test_trtllm_gen_bf16_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    routing_method_type: RoutingMethodType,
):
    """Test Bf16 scale routed MoE matches standard routing."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)

    # Generate random routing logits for reference
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    # Generate random hidden states in FP8
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )

    # Generate weights
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.bfloat16)

    gemm1_weights_shuffled = []
    gemm2_weights_shuffled = []
    for i in range(num_experts):
        tmp_weights1 = shuffle_matrix_a(gemm1_weights[i].view(torch.uint8), 64)
        tmp_weights2 = shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), 64)
        block_k = 128
        gemm1_weights_shuffled.append(convert_to_block_layout(tmp_weights1, block_k))
        gemm2_weights_shuffled.append(convert_to_block_layout(tmp_weights2, block_k))
    gemm1_weights = torch.stack(gemm1_weights_shuffled).view(torch.bfloat16)
    gemm2_weights = torch.stack(gemm2_weights_shuffled).view(torch.bfloat16)

    # Run reference with routing_logits
    reference_output = trtllm_bf16_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
    ).to(torch.float)

    # Compute routing using reference implementation
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights_ref = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights_ref = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, expert_weights_ref = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_ref.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # Pack topk_ids and expert_weights into single tensor
    # Format: (expert_id << 16) | (weight_bf16.view(int16))
    packed_topk_ids = (topk_ids << 16) | expert_weights.view(torch.int16).to(
        torch.int32
    )

    # Run with pre-computed routing (packed format)
    output = trtllm_bf16_routed_moe(
        topk_ids=packed_topk_ids,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
    ).to(torch.float)

    mask = torch.isclose(output, reference_output, rtol=1e-2, atol=1e-2)

    # mismatch percentage
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"

from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit import setup_cubin_loader

@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("hidden_size", [3072])
@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize(
    "global_expert_num,local_expert_num,local_expert_offset",
    [
        (32,32,0),
        # (128,32,32),
    ],
)
def test_trtllm_permute(
    tile_size: int,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    global_expert_num: int,
    local_expert_num: int,
    local_expert_offset: int,
):
    torch.random.manual_seed(42)
    assert local_expert_offset + local_expert_num <= global_expert_num, "Invalid expert configuration"
    module = gen_trtllm_gen_fused_moe_sm100_module()
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))
    # topk_ids = torch.randint(0, global_expert_num, (num_tokens, top_k), device='cuda', dtype=torch.int32)
    routing_logits = torch.rand(num_tokens, global_expert_num, device='cuda').to(
        torch.bfloat16
    )
    permute_info, expert_weights_ = routing_reference_renormalize(
        routing_logits, top_k, global_expert_num, tile_size
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_.view(num_tokens, global_expert_num)[
        torch.arange(num_tokens, device='cuda').unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    expandedTokenIdxToPermutedIdxRef = permute_info["expandedTokenIdxToPermutedIdx"].to(torch.int32)
    print(f'{topk_ids=}')
    print(f'{expandedTokenIdxToPermutedIdxRef.view(num_tokens, top_k)=}')
    hidden_states = torch.randn((num_tokens, hidden_size), device='cuda', dtype=torch.bfloat16)

    expert_weights_int16 = expert_weights.to(torch.bfloat16).view(torch.int16)
    packed_tensor = (topk_ids << 16) | (expert_weights_int16.to(torch.int32))

    perExpertAmax = torch.zeros((global_expert_num, ), device='cuda', dtype=torch.float)
    (
        totalNumPaddedTokens,
        expandedTokenIdxToPermutedIdx,
        ctaIdxXyToBatchIdx,
        ctaIdxXyToMnLimit,
        numNonExitingCtas,
        permutedHiddenStates,
    ) = moe_op.trtllm_permute(
        tile_size,
        top_k,
        global_expert_num,
        local_expert_num,
        local_expert_offset,
        packed_tensor,
        hidden_states,
        perExpertAmax,
    )
    totalNumPaddedTokens = torch.from_dlpack(totalNumPaddedTokens)
    expandedTokenIdxToPermutedIdx = torch.from_dlpack(expandedTokenIdxToPermutedIdx)
    ctaIdxXyToBatchIdx = torch.from_dlpack(ctaIdxXyToBatchIdx)
    ctaIdxXyToMnLimit = torch.from_dlpack(ctaIdxXyToMnLimit)
    numNonExitingCtas = torch.from_dlpack(numNonExitingCtas)
    permutedHiddenStates = torch.from_dlpack(permutedHiddenStates)

    print(f'{totalNumPaddedTokens=}')
    print(f'{expandedTokenIdxToPermutedIdx.view(num_tokens, top_k)=}')
    print(f'{ctaIdxXyToBatchIdx=}')
    print(f'{ctaIdxXyToMnLimit=}')
    print(f'{numNonExitingCtas=}')
    print(f'{permutedHiddenStates.shape=}')
    indices = expandedTokenIdxToPermutedIdx.view(num_tokens, top_k)[:, 0]
    print(f'{hidden_states[:10, :10]=}')
    print(f'{permutedHiddenStates[indices[:10], :10]=}')
    torch.testing.assert_close(hidden_states, permutedHiddenStates[indices, :], atol=1e-2, rtol=1e-2)
    print(f'{perExpertAmax=}')


@pytest.mark.parametrize("tile_size", [128])
@pytest.mark.parametrize("hidden_size", [3072])
@pytest.mark.parametrize("intermediate_size", [2048])
@pytest.mark.parametrize("num_tokens", [16])
@pytest.mark.parametrize("top_k", [8])
@pytest.mark.parametrize("expert_num", [32])
def test_trtllm_nvfp4_batched_gemm(
    tile_size: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    expert_num: int,
):
    device = torch.device('cuda:0')
    torch.random.manual_seed(42)
    module = gen_trtllm_gen_fused_moe_sm100_module()
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))
    routing_logits = torch.rand(num_tokens, expert_num, device='cuda').to(
        torch.bfloat16
    )
    permute_info, expert_weights_ = routing_reference_renormalize(
        routing_logits, top_k, expert_num, tile_size
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_.view(num_tokens, expert_num)[
        torch.arange(num_tokens, device='cuda').unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    packed_tensor = (topk_ids << 16) | (expert_weights.view(torch.int16).to(torch.int32))

    hidden_states = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
    w13 = torch.randn((expert_num, intermediate_size * 2, hidden_size), device=device, dtype=torch.bfloat16)
    w2 = torch.randn((expert_num, hidden_size, intermediate_size), device=device, dtype=torch.bfloat16)

    # generate routing data and permute the hidden states
    (
        totalNumPaddedTokens,
        expandedTokenIdxToPermutedIdx,
        ctaIdxXyToBatchIdx,
        ctaIdxXyToMnLimit,
        numNonExitingCtas,
        permutedHiddenStates,
    ) = moe_op.trtllm_permute(
        tile_size,
        top_k,
        expert_num,
        expert_num,
        0,
        packed_tensor,
        hidden_states,
        None,  # perExpertAmax
    )
    totalNumPaddedTokens = torch.from_dlpack(totalNumPaddedTokens)
    expandedTokenIdxToPermutedIdx = torch.from_dlpack(expandedTokenIdxToPermutedIdx)
    ctaIdxXyToBatchIdx = torch.from_dlpack(ctaIdxXyToBatchIdx)
    ctaIdxXyToMnLimit = torch.from_dlpack(ctaIdxXyToMnLimit)
    numNonExitingCtas = torch.from_dlpack(numNonExitingCtas)
    permutedHiddenStates = torch.from_dlpack(permutedHiddenStates)
    histogram = torch.histc(topk_ids.float().flatten(), bins=expert_num, min=0, max=expert_num-1).int()
    print(f'{topk_ids=}')
    print(f'{histogram=}')
    print(f'{totalNumPaddedTokens=}')
    print(f'{expandedTokenIdxToPermutedIdx.view(num_tokens, top_k)=}')
    print(f'{ctaIdxXyToBatchIdx=}')
    print(f'{ctaIdxXyToMnLimit=}')
    print(f'{numNonExitingCtas=}')
    print(f'{permutedHiddenStates.shape=}')
    indices = expandedTokenIdxToPermutedIdx.view(num_tokens, top_k)[:, 0]
    print(f'{hidden_states[:10, :10]=}')
    print(f'{permutedHiddenStates[indices[:10], :10]=}')

    num_tokens_padded = permutedHiddenStates.shape[0]
    max_num_ctas = ctaIdxXyToBatchIdx.shape[0]

    print(f'{num_tokens_padded=}')
    print(f'{max_num_ctas=}')

    # quantize the weight and permuted hidden states
    # TODO: use perExpertAmax to quantize
    hidden_states_global_scale = (448 * 6) / hidden_states.float().abs().nan_to_num().max()
    w13_global_scale = (448 * 6) / w13.float().abs().nan_to_num().max()
    hidden_states_fp4, hidden_states_scales = fp4_quantize(
        permutedHiddenStates,
        hidden_states_global_scale,
        is_sf_swizzled_layout=True,
    )

    w13_fp4_shuffled = []
    w13_scales_shuffled = []
    for i in range(expert_num):
        w13_fp4, w13_scales = fp4_quantize(
            w13[i],
            w13_global_scale,
            is_sf_swizzled_layout=False, # shuffle_matrix_sf_a will swizzle it to 128x4
        )
        w13_fp4 = shuffle_matrix_a(w13_fp4, 128).reshape(intermediate_size*2, hidden_size // 2)
        w13_scales = shuffle_matrix_sf_a(w13_scales, 128).reshape(intermediate_size*2, hidden_size // 16)
        w13_fp4_shuffled.append(w13_fp4)
        w13_scales_shuffled.append(w13_scales)
    w13_fp4 = torch.stack(w13_fp4_shuffled)
    w13_scales = torch.stack(w13_scales_shuffled)
    fc1_output = torch.empty((num_tokens_padded, intermediate_size * 2), device=device, dtype=torch.bfloat16)
    
    fc1_output_scales = torch.ones((expert_num, ), device=device, dtype=torch.float32) * hidden_states_global_scale * w13_global_scale


    moe_op.trtllm_nvfp4_batched_gemm(
        num_tokens,
        expert_num,
        top_k,
        tile_size,
        -1, # tactic
        False, # fuseActivation
        True, # enablePDL
        hidden_states_fp4,
        hidden_states_scales,
        w13_fp4,
        w13_scales,
        None, # bias
        None, # swiglu alpha
        None, # swiglu beta
        None, # swiglu clamp limit
        max_num_ctas,
        totalNumPaddedTokens,
        numNonExitingCtas,
        ctaIdxXyToBatchIdx,
        ctaIdxXyToMnLimit,
        None, # routeMap
        None, # tokenScales
        fc1_output_scales, # outputScales
        None, # outputScalesGate
        fc1_output
    )

    fc1_act = torch.nn.functional.silu(fc1_output[:, :intermediate_size]) * fc1_output[:, intermediate_size:]
    print(f'{fc1_act.shape=}')
    # quantize the w2 and fc1_act
    fc1_act_global_scale = (448 * 6) / fc1_act.float().abs().nan_to_num().max()
    w2_global_scale = (448 * 6) / w2.float().abs().nan_to_num().max()
    fc1_act_fp4, fc1_act_scales = fp4_quantize(
        fc1_act,
        fc1_act_global_scale.to(device),
        is_sf_swizzled_layout=True,
    )
    w2_fp4_shuffled = []
    w2_scales_shuffled = []
    for i in range(expert_num):
        w2_fp4_i, w2_scales_i = fp4_quantize(
            w2[i],
            w2_global_scale.to(device),
            is_sf_swizzled_layout=False, # shuffle_matrix_sf_a will swizzle it to 128x4
        )
        w2_fp4_i = shuffle_matrix_a(w2_fp4_i, 128).reshape(hidden_size, intermediate_size // 2)
        w2_scales_i = shuffle_matrix_sf_a(w2_scales_i, 128).reshape(hidden_size, intermediate_size // 16)
        w2_fp4_shuffled.append(w2_fp4_i)
        w2_scales_shuffled.append(w2_scales_i)
    w2_fp4 = torch.stack(w2_fp4_shuffled)
    w2_scales = torch.stack(w2_scales_shuffled)
    fc2_output = torch.empty((num_tokens_padded, hidden_size), device=device, dtype=torch.bfloat16)
    fc2_output_scales = torch.ones((expert_num, ), device=device, dtype=torch.float32) * fc1_act_global_scale * w2_global_scale
    moe_op.trtllm_nvfp4_batched_gemm(
        num_tokens,
        expert_num,
        top_k,
        tile_size,
        -1, # tactic
        False, # fuseActivation
        True, # enablePDL
        fc1_act_fp4,
        fc1_act_scales,
        w2_fp4,
        w2_scales,
        None, # bias
        None, # swiglu alpha
        None, # swiglu beta
        None, # swiglu clamp limit
        max_num_ctas,
        totalNumPaddedTokens,
        numNonExitingCtas,
        ctaIdxXyToBatchIdx,
        ctaIdxXyToMnLimit,
        None, # routeMap
        None, # tokenScales
        fc2_output_scales, # outputScales
        None, # outputScalesGate
        fc2_output
    )