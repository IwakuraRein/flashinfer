#!/usr/bin/env python3
"""Kimi-K3 routed latent-MoE comparison, including CuTe-DSL Mega-MoE.

Preview (no torch import, CUDA initialization, or benchmarks):
    python benchmarks/bench_kimi_k3_moe_sweep.py
Run later:
    python benchmarks/bench_kimi_k3_moe_sweep.py --run --output kimi_k3_moe.csv

Uses the unified runners and CUPTI timing pattern from
benchmarks/routines/moe.py, behind flashinfer_benchmark.py. Both backends
receive identical synthetic inputs, precomputed routing, and local weights.
Weight/activation quantization, router top-k, latent projections/norm, shared
experts, and communication are excluded from timing. Dispatch, expert GEMMs,
activation, and local weighted combine are timed. Activation defaults to the
model config's hidden_act (SiTU for Kimi-K3), with its configured gate/linear
scales; unsupported combinations are explicitly skipped. Use --activation
swiglu only to request an alternative shape comparison.

Batch means GLOBAL input tokens, not tokens per EP rank. TP8 and EP8 are
separate configurations, not TP8*EP8: TP8 keeps all experts and divides the
intermediate dimension by eight; EP8 keeps one eighth of the experts and
the full intermediate dimension. EP routing remains global top-16: do not
divide batch or top-k by eight, or route every token to local experts.
Only the selected local expert pairs are computed on the simulated rank.
This is a compute simulation, not distributed end-to-end latency.

Mega-MoE supports NVFP4 x NVFP4 with SiTU. TP8 runs the local intermediate=384
shape with MEGA_NO_DIST=1 (no TP all-reduce). EP8 launches eight processes via
torch.distributed.run on this node, splitting the GLOBAL batch exactly across
ranks, including empty input ranks for batch=1. Run this script with python,
not torchrun: it launches its own workers. CUDA_VISIBLE_DEVICES must expose
eight GPUs for EP8. Mega timing includes input quantization/staging and fused
dispatch/compute/combine, so it is labeled separately from compute-only rows.
Mega uses fixed-count, lockstep CUDA-event timing of graph replays and reports
the maximum rank latency per iteration; it does not use adaptive CUPTI loops.
CuTe-DSL Mega-MoE has no MXFP4 x MXFP8 mode; those rows remain skipped.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CONFIG = Path("/home/scratch.trt_llm_data_ci/llm-models/Kimi-K3/config.json")
BATCHES = (1, 8, 32, 128, 256, 512)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run", action="store_true", help="Execute; default is preview only."
    )
    parser.add_argument("--output", type=Path, default=Path("kimi_k3_moe.csv"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCHES)
    parser.add_argument(
        "--parallel", nargs="+", choices=("tp8", "ep8"), default=("tp8", "ep8")
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=("mxfp4_mxfp8", "nvfp4_nvfp4"),
        default=("mxfp4_mxfp8", "nvfp4_nvfp4"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("trtllm-gen", "cute-dsl", "mega-moe"),
        default=("trtllm-gen", "cute-dsl", "mega-moe"),
    )
    parser.add_argument(
        "--activation",
        choices=("swiglu", "situ"),
        default=None,
        help="Override the activation (default: hidden_act from the model config).",
    )
    parser.add_argument("--ep-rank", type=int, choices=range(8), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--use-cuda-events", action="store_true")
    parser.add_argument("--mega-worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--mega-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch_sizes) <= 0 or args.num_iters <= 0 or args.dry_run_iters < 0:
        parser.error(
            "batch sizes and num-iters must be positive; dry-run-iters nonnegative"
        )
    return args


def cases(args, model):
    hidden = model["routed_expert_hidden_size"]
    intermediate = model["moe_intermediate_size"]
    experts = model["num_experts"]
    if intermediate % 8 or experts % 8:
        raise ValueError(
            "TP8/EP8 requires intermediate size/expert count divisible by eight"
        )
    for parallel, quant, batch, backend in itertools.product(
        args.parallel, args.quant, args.batch_sizes, args.backends
    ):
        reason = ""
        if backend == "mega-moe" and quant != "nvfp4_nvfp4":
            reason = "CuTe-DSL Mega-MoE has no MXFP4 x MXFP8 mode"
        elif (
            backend == "cute-dsl"
            and quant == "mxfp4_mxfp8"
            and args.activation == "situ"
        ):
            reason = "CuTe-DSL W4A8 does not support SiTU"
        yield dict(
            parallel=parallel,
            quant=quant,
            batch_global=batch,
            backend=backend,
            activation=args.activation,
            timing_scope=(
                "staging+dispatch+compute+combine"
                if backend == "mega-moe"
                else "compute_only"
            ),
            world_size=8 if backend == "mega-moe" and parallel == "ep8" else 1,
            hidden_size=hidden,
            intermediate_size=intermediate // 8 if parallel == "tp8" else intermediate,
            num_experts=experts,
            top_k=model["num_experts_per_token"],
            local_num_experts=experts if parallel == "tp8" else experts // 8,
            local_expert_offset=(
                0
                if parallel == "tp8" or backend == "mega-moe"
                else args.ep_rank * (experts // 8)
            ),
            status="skipped" if reason else "planned",
            reason=reason,
        )


def measure(args, model, case):
    # Imports stay inside the explicitly requested execution path.
    import torch
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoEFinalizeConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SiTU,
        SwiGLU,
        TrtllmFp4Config,
    )
    from flashinfer.testing import bench_gpu_time

    device = torch.device("cuda")
    h, i, e, k, m = (
        case[name]
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "batch_global",
        )
    )
    local, offset = case["local_num_experts"], case["local_expert_offset"]
    config_type = TrtllmFp4Config if case["backend"] == "trtllm-gen" else CuteDslConfig
    quant = (
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        if case["quant"] == "mxfp4_mxfp8"
        else QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    )
    activation = (
        SiTU(
            gate_scale=model["activation_situ_beta"],
            linear_scale=model["activation_situ_linear_beta"],
        )
        if args.activation == "situ"
        else SwiGLU()
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=k),
        quant=quant,
        experts=ExpertConfig(
            intermediate_size=i, local_num_experts=local, local_expert_offset=offset
        ),
        activation=activation,
        backend=BackendOptions(candidates=(config_type(),)),
        finalize=MoEFinalizeConfig(do_finalize=True, use_fused_finalize=True),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=m),
    )
    layer = MoELayer(config, device=device)
    if len(layer.runners) != 1:
        raise RuntimeError(f"Expected one explicit backend, got {len(layer.runners)}")
    runner = layer.runners[0]

    # Identical RNG order for both backends. Synthetic sigmoid routing without
    # a learned correction bias; the config supplies no bias tensor.
    torch.manual_seed(args.seed)
    logits = torch.randn(m, e, device=device, dtype=torch.float32)
    scores = logits.sigmoid()
    weights, ids = scores.topk(k, dim=-1, sorted=False)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights *= model.get("routed_scaling_factor", 1.0)
    # Packed TRTLLM routing stores weights as BF16; give CuTe the same values.
    weights = weights.bfloat16().float()
    pairs = int(((ids >= offset) & (ids < offset + local)).sum().item())
    x = torch.randn(m, h, device=device, dtype=torch.bfloat16) / 10
    xq, xs = TrtllmFp4Config.prepare_activations(x, quant=quant)
    act_pack = MoEActivationPack(
        hidden_states_q=xq,
        hidden_states_scale=xs,
        topk_ids=ids.int(),
        topk_weights=weights,
    )
    w1 = torch.randn(local, 2 * i, h, device=device, dtype=torch.bfloat16) / 10
    w2 = torch.randn(local, h, i, device=device, dtype=torch.bfloat16) / 10
    view = config_type.prepare_weights(
        w1,
        w2,
        quant=quant,
        num_local_experts=local,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
        device=device,
    )
    del w1, w2, x, logits, scores
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(runner.backend_key, view)
    inputs = runner.pack_inputs(act_pack, weight_pack)
    launch_kwargs = runner.launch_kwargs_for(inputs)
    with autotune(not args.no_autotune):
        _, tactic = AutoTuner.get().choose_one(
            custom_op=f"moe_{runner.backend_key}",
            runners=[runner],
            tuning_config=runner.tuning_config_for(inputs),
            inputs=inputs,
            **launch_kwargs,
        )

    def run(*profile_inputs):
        return runner.forward(list(profile_inputs), tactic=tactic, **launch_kwargs)

    runner.forward(inputs, tactic=tactic, do_preparation=True, **launch_kwargs)
    torch.cuda.synchronize()
    times = bench_gpu_time(
        run,
        input_args=tuple(inputs),
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.num_iters,
        enable_cupti=not args.use_cuda_events,
        use_cuda_graph=True,
        num_iters_within_graph=1,
        cold_l2_cache=True,
        sleep_after_run=False,
    )
    median = float(statistics.median(times))
    return dict(
        status="ok",
        local_pairs=pairs,
        median_ms=median,
        std_ms=float(statistics.pstdev(times)),
        tflops=6 * pairs * h * i / (median * 1e9),
        gpu=torch.cuda.get_device_name(),
        runner=runner.backend_key,
    )


def launch_mega(args, case):
    """Isolate symmetric memory/runtime state in a fresh worker per case."""
    with tempfile.TemporaryDirectory(prefix="kimi-mega-") as directory:
        result = Path(directory) / "result.json"
        command = [sys.executable]
        if case["world_size"] > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=8",
            ]
        command += [
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "--activation",
            args.activation,
            "--num-iters",
            str(args.num_iters),
            "--dry-run-iters",
            str(args.dry_run_iters),
            "--seed",
            str(args.seed),
            "--mega-worker-case",
            json.dumps(case),
            "--mega-result",
            str(result),
        ]
        if args.no_autotune:
            command.append("--no-autotune")
        env = os.environ.copy()
        env["MEGA_NO_DIST"] = "1" if case["world_size"] == 1 else "0"
        env["MEGA_SINGLE_GPU_GLOO"] = "0"
        subprocess.run(command, env=env, check=True)
        return json.loads(result.read_text())


def mega_worker(args, model, case):
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    world = case["world_size"]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.device_count() < world:
        raise RuntimeError(f"Mega {case['parallel']} requires {world} visible GPUs")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    def barrier():
        if world > 1:
            dist.barrier()

    h, i, e, k, m = (
        case[name]
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "batch_global",
        )
    )
    # Contiguous balanced partition preserves the exact global batch. For G=1,
    # rank zero owns the token; all eight ranks still compute their experts.
    begin = rank * (m // world) + min(rank, m % world)
    count = m // world + int(rank < m % world)
    capacity = (m + world - 1) // world
    local = e // world
    torch.manual_seed(args.seed)
    scores = torch.randn(m, e, device="cuda", dtype=torch.float32).sigmoid()
    weights, ids = scores.topk(k, dim=-1, sorted=False)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = (weights * model.get("routed_scaling_factor", 1.0)).bfloat16().float()
    pairs = int(((ids >= rank * local) & (ids < (rank + 1) * local)).sum().item())
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16) / 10
    tensors = MoEEpTensors(
        hidden_states=x[begin : begin + count].contiguous(),
        topk_ids=ids[begin : begin + count].contiguous(),
        topk_weights=weights[begin : begin + count].contiguous(),
    )
    # Rank-specific local expert weights; rank zero uses the regular sweep's RNG state.
    if rank:
        torch.manual_seed(args.seed + rank)
    w1 = torch.randn(local, 2 * i, h, device="cuda", dtype=torch.bfloat16) / 10
    w2 = torch.randn(local, h, i, device="cuda", dtype=torch.bfloat16) / 10
    kernel = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=i,
        top_k=k,
        activation=args.activation,
        situ_beta=model["activation_situ_beta"] if args.activation == "situ" else None,
        situ_linear_beta=(
            model["activation_situ_linear_beta"] if args.activation == "situ" else None
        ),
        # Apply router weights after the nonlinear expert, matching regular MoE.
        apply_topk_in_fc1=False,
        knobs=None if args.no_autotune else "auto",
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world, rank=rank, auto_bootstrap=world > 1
        ),
        fleet_params=FleetParams(
            num_experts=e, max_tokens_per_rank=capacity, token_hidden_size=h
        ),
        weights=MoEWeightPack(w13=w1, w2=w2),
        backend=MegaConfig(
            megakernel=kernel, quantize_input=True, preprocess_weights=True
        ),
    )
    # No rank-local adaptive benchmark loops: the kernel has collective barriers.
    layer.warmup(tensors)
    barrier()
    layer.forward(tensors)
    torch.cuda.synchronize()
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer.forward(tensors, return_workspace_view=True)
    barrier()
    for _ in range(args.dry_run_iters):
        graph.replay()
        torch.cuda.synchronize()
        barrier()

    # Explicit L2 eviction outside the timed range on every rank, every iteration.
    props = torch.cuda.get_device_properties(local_rank)
    flush = torch.empty(
        max(256 * 1024**2, 2 * props.L2_cache_size), dtype=torch.uint8, device="cuda"
    )
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    samples = []
    for _ in range(args.num_iters):
        flush.zero_()
        torch.cuda.synchronize()
        barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        barrier()
    all_samples = torch.tensor(samples, device="cuda", dtype=torch.float64)
    if world > 1:
        dist.all_reduce(all_samples, op=dist.ReduceOp.MAX)
    times = all_samples.cpu().tolist()
    pair_counts = [None] * world
    if world > 1:
        dist.all_gather_object(pair_counts, pairs)
    else:
        pair_counts[0] = pairs
    median = float(statistics.median(times))
    if rank == 0:
        args.mega_result.write_text(
            json.dumps(
                dict(
                    status="ok",
                    local_pairs=pair_counts[0],
                    pairs_per_rank=pair_counts,
                    tokens_per_rank=[
                        m // world + int(r < m % world) for r in range(world)
                    ],
                    median_ms=median,
                    std_ms=float(statistics.pstdev(times)),
                    # Average per-rank throughput; latency is the maximum across ranks.
                    tflops=6 * (m * k / world) * h * i / (median * 1e9),
                    gpu=torch.cuda.get_device_name(),
                    runner=kernel.kernel_name,
                    timer="cuda_event_rank_max",
                )
            )
        )
    del graph, output
    torch.cuda.synchronize()
    barrier()
    layer.destroy()
    if world > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    model = json.loads(args.config.read_text())["text_config"]
    if args.activation is None:
        args.activation = model["hidden_act"]
    if args.activation not in ("swiglu", "situ"):
        raise ValueError(f"Unsupported model activation: {args.activation!r}")
    if model["moe_router_activation_func"] != "sigmoid" or not model["moe_renormalize"]:
        raise ValueError("This script expects sigmoid routing with renormalization")
    if args.mega_worker_case is not None:
        mega_worker(args, model, json.loads(args.mega_worker_case))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Launch this script with python, not torchrun; it manages its own workers"
        )
    plan = list(cases(args, model))
    print("Global batch; TP8/EP8 compute simulations, plus real EP8 for Mega-MoE.")
    print(f"Activation: {args.activation} (model native: {model['hidden_act']}).")
    print(
        "Shared experts, latent projections/norm and router top-k excluded. "
        "Mega-MoE additionally times staging and communication (see timing_scope)."
    )
    for case in plan:
        print(json.dumps(case))
    if not args.run:
        print("Preview only. Add --run to execute benchmarks.")
        return

    fields = list(plan[0]) + [
        "local_pairs",
        "median_ms",
        "std_ms",
        "tflops",
        "gpu",
        "runner",
        "timer",
        "tokens_per_rank",
        "pairs_per_rank",
    ]
    # Exclusive creation protects results from accidental overwrite.
    with args.output.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for case in plan:
            if case["status"] != "skipped":
                try:
                    case.update(
                        launch_mega(args, case)
                        if case["backend"] == "mega-moe"
                        else measure(args, model, case)
                    )
                except Exception as error:
                    case.update(
                        status="error", reason=f"{type(error).__name__}: {error}"
                    )
                    writer.writerow(case)
                    output.flush()
                    raise  # Preserve the failure; do not call it unsupported.
                finally:
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
            writer.writerow(case)
            output.flush()
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
