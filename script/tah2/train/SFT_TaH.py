import argparse
import os
from typing import Dict

import torch
import yaml
from tah2.train.simple_model import (
    load_model_and_tokenizer,
    maybe_enable_gradient_checkpointing,
)
from tah2.utils.data_prepare import preprocess_dataset
from tah2.utils.modeling import set_all_seeds


def load_config(config_path: str) -> Dict:
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main_tp(
    config,
    tp_size: int,
    vocab_parallel: bool = True,
    output_dir_override: str | None = None,
    hsdp_replicate: int = 1,
):
    """Torch-native FSDP2(dp) × DTensor-TP(tp) training (torchrun launch).

    Usage:
        torchrun --nproc_per_node <world> script/tah2/train/SFT_TaH.py \
            --config <recipe.yaml> --tp <tp_size>

    World size = dp * tp. TP=1 uses the same native FSDP2 loop without TP
    transforms; TP>=2 additionally applies tensor/vocab parallelism. The loop
    lives in tah2.train.simple_loop.train_model_tp.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from tah2.train.simple_loop import train_model_tp
    from tah2.train.simple_runtime import DistShim
    from tah2.train.tp import (
        apply_fsdp,
        apply_tensor_parallel,
        scope_tah_sync_group_solo,
    )

    model_config, data_config, training_config = (
        config["model"],
        config["data"],
        config["training"],
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tp ({tp_size})"
        )
    dp_size = world_size // tp_size
    # HSDP (--hsdp_replicate, a launch-topology knob like --tp, not a recipe
    # field): dp splits into (replicate, shard). Weights are sharded only
    # within a shard group (keep it intra-node); gradients are all-reduced
    # across replicate groups -- inter-node DDP semantics, which cuts
    # cross-node traffic from per-micro-batch weight AG + full-grad RS to one
    # sharded-grad AR. fully_shard receives the 2-D mesh and does the rest.
    if hsdp_replicate > 1:
        if dp_size % hsdp_replicate:
            raise ValueError(
                f"dp ({dp_size}) must be divisible by hsdp_replicate ({hsdp_replicate})"
            )
        mesh = init_device_mesh(
            "cuda",
            (hsdp_replicate, dp_size // hsdp_replicate, tp_size),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        dp_mesh = mesh["dp_replicate", "dp_shard"]
        # Register a flattened "dp" dim so every mesh["dp"] / get_group("dp")
        # consumer (loss sync, label generator, data sharding) sees the full
        # data-parallel group exactly as in the flat layout. _flatten is
        # underscore-private but is the sanctioned HSDP idiom (torchtitan).
        dp_mesh._flatten(mesh_dim_name="dp")
    else:
        mesh = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = mesh["dp"]
    # Isolate routing collectives from FSDP/TP communication.
    tah_sync_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")

    shim = DistShim(device)
    shim.print(
        f"world={world_size} dp={dp_size} tp={tp_size}"
        + (f" (hsdp replicate={hsdp_replicate})" if hsdp_replicate > 1 else "")
    )

    # fp32 load: FSDP2 keeps sharded fp32 master params, bf16 compute via
    # MixedPrecisionPolicy (same semantics as accelerate-FSDP / DeepSpeed).
    if str(model_config.get("torch_dtype", "")).lower() not in ("float32", "fp32"):
        shim.print("[fsdp] overriding model.torch_dtype -> float32 (master weights)")
        model_config = {**model_config, "torch_dtype": "float32"}

    model, tokenizer = load_model_and_tokenizer(training_config, model_config, shim)
    model = model.to(device)
    model.tah_sync_group = tah_sync_group
    model.dp_group = mesh.get_group("dp")
    if getattr(model, "iter_label_generator", None) is not None:
        model.iter_label_generator.tah_sync_group = tah_sync_group
        model.iter_label_generator.dp_group = model.dp_group
    if tp_size == 1:
        shim.print("[tp] tp=1: tensor/vocab parallel disabled")
        # Each rank is a full DP replica, so the recurrent iter-sync collectives
        # have no cross-rank meaning; left on the world group they are the
        # tp>1 desync hazard documented in tp.py. Make them local no-ops.
        scope_tah_sync_group_solo(model, dist.get_rank(), world_size)
    else:
        model = apply_tensor_parallel(
            model, mesh["tp"], vocab_parallel=vocab_parallel
        )
    maybe_enable_gradient_checkpointing(model, training_config, shim)
    # Keep the conservative default for recurrent models.
    layer_wrap_override = training_config.get("fsdp_layer_wrap", None)
    layer_wrap = (
        int(model_config.get("max_iter", 1) or 1) == 1
        if layer_wrap_override is None
        else bool(layer_wrap_override)
    )
    # Per-layer FSDP requires aligned recurrent graphs.
    model._tah_require_nonempty_recurrent_graph = layer_wrap
    model = apply_fsdp(model, dp_mesh, layer_wrap=layer_wrap)
    model.train()

    local_numel = sum(
        p.to_local().numel() if isinstance(p, DTensor) else p.numel()
        for p in model.parameters()
    )
    global_numel = sum(p.numel() for p in model.parameters())
    shim.print(
        f"params: global={global_numel / 1e9:.3f}B "
        f"local_shard={local_numel / 1e9:.3f}B "
        f"(fsdp layer_wrap={layer_wrap}, fp32 master + bf16 compute)"
    )

    train_dataset, eval_dataset = preprocess_dataset(data_config, tokenizer, shim)

    output_dir = output_dir_override or os.path.join(
        data_config.get("output_dir", "output/tp"), f"tp{tp_size}_dp{dp_size}"
    )
    os.makedirs(output_dir, exist_ok=True)
    if shim.is_main_process:
        config_save_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        shim.print(f"Configuration saved to: {config_save_path}")

    resume_path = None
    if training_config.get("resume_from_ckpt", False) and (
        "tah_model_path" in model_config
    ):
        resume_path = model_config["tah_model_path"]
        shim.print(f"Resume-from-ckpt enabled: {resume_path}")

    history = train_model_tp(
        model=model,
        tokenizer=tokenizer,
        processed_train_dataset=train_dataset,
        output_dir=output_dir,
        training_config=training_config,
        data_config=data_config,
        run_config=config,
        mesh=mesh,
        shim=shim,
        processed_eval_dataset=eval_dataset,
        resume_from_checkpoint_path=resume_path,
    )

    if bool(training_config.get("skip_final_save", False)):
        shim.print("skip_final_save=true, skipping final model save.")
    else:
        from tah2.train.simple_loop import save_model_tp

        save_model_tp(model, tokenizer, os.path.join(output_dir, "final_model"), shim)

    dist.barrier()
    dist.destroy_process_group(tah_sync_group)
    dist.destroy_process_group()
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a causal language model with configuration file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Torch-native tensor-parallel size: 1 = FSDP2 only (default); "
        ">=2 additionally shards attention/MLP/vocab.",
    )
    parser.add_argument(
        "--no_vocab_parallel",
        action="store_true",
        help="TP mode only: disable vocab (embedding/lm_head) sharding; "
        "only shard backbone linears.",
    )
    parser.add_argument(
        "--hsdp_replicate",
        type=int,
        default=1,
        help="TP mode only: split dp into (replicate, shard) for HSDP. "
        "Keep the shard group intra-node; >1 replicates weights across "
        "groups and all-reduces sharded grads between them (inter-node DDP).",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    set_all_seeds(int(config.get("seed", 42)))

    if args.tp < 1:
        parser.error("--tp must be >= 1")
    if args.hsdp_replicate < 1:
        parser.error("--hsdp_replicate must be >= 1")
    main_tp(
        config,
        tp_size=args.tp,
        vocab_parallel=not args.no_vocab_parallel,
        output_dir_override=args.output_dir,
        hsdp_replicate=args.hsdp_replicate,
    )
