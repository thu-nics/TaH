import itertools
import math
import os
import shutil
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from tah2.train.data_collator import CustomTaHDataCollator
from tah2.train.simple_runtime import (
    MetricLogger,
    compute_eval_global_batch_size,
    compute_eval_num_items,
    sync_training_state,
)


def _to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().float().item()
        return value.detach().float().mean().item()
    return value


def _format_metrics(metrics: Dict, step: int) -> Dict:
    formatted: Dict = {"step": int(step)}
    for key, value in metrics.items():
        if key == "epoch":
            continue
        value = _to_scalar(value)
        if isinstance(value, float):
            formatted[key] = float(f"{value:.4g}")
        elif isinstance(value, int):
            formatted[key] = int(value)
        else:
            try:
                formatted[key] = float(f"{float(value):.4g}")
            except Exception:
                formatted[key] = value
    return formatted


def _prefix_metrics(metrics: Dict, prefix: str) -> Dict:
    prefixed: Dict = {}
    for key, value in metrics.items():
        if key == "step":
            prefixed[key] = value
            continue

        base = key
        if base in ("train_loss", "eval_loss"):
            base = "loss"
        elif base.startswith("train_"):
            base = base[len("train_") :]
        elif base.startswith("eval_"):
            base = base[len("eval_") :]

        prefixed[f"{prefix}/{base}"] = value
    return prefixed


def _terminal_metrics(metrics: Dict) -> Dict:
    """Compact terminal view: keep values unchanged, strip train/eval key prefixes."""
    compact: Dict = {}
    for key, value in metrics.items():
        if isinstance(key, str):
            if key.startswith("train/"):
                compact[key[len("train/") :]] = value
                continue
            if key.startswith("eval/"):
                compact[key[len("eval/") :]] = value
                continue
        compact[key] = value
    return compact


def build_optimizer(
    model,
    training_config: Dict,
    separate_lr_config: Optional[Dict[str, float]],
    accelerator,  # DistShim (print/device duck-type)
):
    default_lr = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    param_groups = []
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay_parameter_names = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
    decay_parameter_names = {
        name for name in decay_parameter_names if "bias" not in name
    }

    if not named_params:
        raise ValueError("No trainable parameters found.")

    assigned_param_names = set()
    if separate_lr_config:
        accelerator.print(f"Using separate learning rates: {separate_lr_config}")

        for component_name, lr in separate_lr_config.items():
            prefix = f"{component_name}."
            decay_params = []
            no_decay_params = []

            for full_name, param in named_params:
                if full_name.startswith(prefix):
                    assigned_param_names.add(full_name)
                    if full_name in decay_parameter_names:
                        decay_params.append(param)
                    else:
                        no_decay_params.append(param)

            if not decay_params and not no_decay_params:
                accelerator.print(
                    f"Warning: component '{component_name}' not found in model trainable parameters."
                )
                continue

            if decay_params:
                param_groups.append(
                    {"params": decay_params, "weight_decay": weight_decay, "lr": lr}
                )
            if no_decay_params:
                param_groups.append(
                    {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
                )

            accelerator.print(
                f"Param groups for '{component_name}' (lr={lr}): "
                f"{len(decay_params)} decay + {len(no_decay_params)} no-decay"
            )

    remaining_decay_params = []
    remaining_no_decay_params = []
    for full_name, param in named_params:
        if full_name in assigned_param_names:
            continue
        if full_name in decay_parameter_names:
            remaining_decay_params.append(param)
        else:
            remaining_no_decay_params.append(param)

    if remaining_decay_params:
        param_groups.append(
            {
                "params": remaining_decay_params,
                "weight_decay": weight_decay,
                "lr": default_lr,
            }
        )
    if remaining_no_decay_params:
        param_groups.append(
            {"params": remaining_no_decay_params, "weight_decay": 0.0, "lr": default_lr}
        )

    accelerator.print(
        f"Param groups for 'default' (lr={default_lr}): "
        f"{len(remaining_decay_params)} decay + {len(remaining_no_decay_params)} no-decay"
    )

    adam_beta1 = float(training_config.get("adam_beta1", 0.9))
    adam_beta2 = float(training_config.get("adam_beta2", 0.999))
    adam_epsilon = float(training_config.get("adam_epsilon", 1e-8))
    optim_name = str(training_config.get("optim", "adamw_torch_fused")).lower()
    use_fused = (
        ("fused" in optim_name)
        and ("cuda" in str(accelerator.device))
        and torch.cuda.is_available()
    )

    optimizer_kwargs = {
        "lr": default_lr,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }

    if use_fused:
        optimizer_kwargs["fused"] = True
        try:
            return torch.optim.AdamW(param_groups, **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            accelerator.print(
                "Warning: fused AdamW is not supported in this environment, fallback to non-fused AdamW."
            )
            return torch.optim.AdamW(param_groups, **optimizer_kwargs)

    return torch.optim.AdamW(param_groups, **optimizer_kwargs)


def create_scheduler(optimizer, training_config: Dict, max_steps: int):
    warmup_ratio = training_config.get("warmup_ratio", 0.0)
    warmup_steps = int(max_steps * warmup_ratio)
    scheduler_type = training_config.get("lr_scheduler_type", "linear")
    scheduler_kwargs = training_config.get("lr_scheduler_kwargs", {}) or {}

    try:
        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            scheduler_specific_kwargs=scheduler_kwargs,
        )
    except TypeError:
        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )


def rotate_checkpoints(
    output_dir: str, save_total_limit: Optional[int], accelerator
):
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoints = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            checkpoints.append(path)

    if len(checkpoints) <= save_total_limit:
        return

    checkpoints.sort(key=os.path.getmtime)
    to_delete = checkpoints[: len(checkpoints) - save_total_limit]
    for path in to_delete:
        accelerator.print(f"Removing old checkpoint due to save_total_limit: {path}")
        shutil.rmtree(path, ignore_errors=True)


def save_model_tp(model, tokenizer, save_dir: str, shim) -> None:
    """Save a sharded TaH model as a normal full bf16 checkpoint (model-only).

    COLLECTIVE — every rank must call it; global rank 0 writes (multi-node
    output dirs must be on the shared filesystem). Side modules (decider /
    updater) are dp-sharded DTensors under FSDP and must be gathered to plain
    tensors too, else the .bin files hold a single rank's shard. Weights are
    cast to bf16 (the serving format); exact fp32 masters live in the DCP
    train state when ``save_only_model: false``.
    """
    import torch.distributed as dist

    from tah2.train.tp import full_base_state_dict, full_plain_state_dict

    full_sd = full_base_state_dict(model, cast_floats_to=torch.bfloat16)
    decider_sd = full_plain_state_dict(
        getattr(model, "iter_decider", None), cast_floats_to=torch.bfloat16
    )
    updater_sd = full_plain_state_dict(
        getattr(model, "input_updater", None), cast_floats_to=torch.bfloat16
    )
    base = getattr(model, "simple_base_model", model)
    inner = getattr(base, "model", base)
    # Tied lm_head: HF's save dedups it automatically only for its own
    # state_dict, not a hand-built one — drop the duplicate (+0.6GB otherwise);
    # from_pretrained re-ties via config.tie_word_embeddings.
    head_weight = getattr(getattr(base, "lm_head", None), "weight", None)
    if head_weight is not None and head_weight is inner.embed_tokens.weight:
        full_sd.pop("lm_head.weight", None)
    # The model trains as fp32 masters; advertise the saved dtype instead.
    base.config.dtype = torch.bfloat16
    base.config.torch_dtype = torch.bfloat16
    if dist.get_rank() == 0:
        model.save_pretrained(
            save_dir,
            state_dict=full_sd,
            decider_state_dict=decider_sd,
            updater_state_dict=updater_sd,
        )
        # HF save_pretrained re-stamps config dtype from the live parameters
        # (fp32 masters), clobbering the bf16 advertisement above; consumers
        # (mini-sglang) then run the whole model in fp32. Fix the serialized
        # config to match the actually-saved bf16 tensors.
        import json

        cfg_path = os.path.join(save_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                saved_cfg = json.load(f)
            for key in ("dtype", "torch_dtype"):
                if key in saved_cfg:
                    saved_cfg[key] = "bfloat16"
            with open(cfg_path, "w") as f:
                json.dump(saved_cfg, f, indent=2, sort_keys=True)
        tokenizer.save_pretrained(save_dir)
        # Save the chat template's turn-ending token as EOS/PAD, without
        # changing the live tokenizer or the training collator's padding.
        if tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            special_ids = set(tokenizer.all_special_ids)
            end_id = next((i for i in reversed(ids) if i in special_ids), None)
            if end_id is not None:
                end_token = tokenizer.convert_ids_to_tokens(end_id)
                updates = {
                    "config.json": {"eos_token_id": end_id, "pad_token_id": end_id},
                    "tokenizer_config.json": {"eos_token": end_token, "pad_token": end_token},
                    "special_tokens_map.json": {"eos_token": end_token, "pad_token": end_token},
                    "generation_config.json": {"eos_token_id": end_id, "pad_token_id": end_id},
                }
                for filename, values in updates.items():
                    path = os.path.join(save_dir, filename)
                    if not os.path.exists(path):
                        continue
                    with open(path, encoding="utf-8") as f:
                        saved = json.load(f)
                    for key, value in values.items():
                        if isinstance(saved.get(key), dict):
                            saved[key]["content"] = value
                        else:
                            saved[key] = value
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(saved, f, indent=2, ensure_ascii=False)
        shim.print(f"checkpoint saved -> {save_dir}")
    del full_sd
    dist.barrier()



def setup_wandb_tp(training_config: Dict, config: Dict, output_dir: str, rank: int):
    if rank != 0 or str(training_config.get("report_to", "none")).lower() != "wandb":
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        print("WANDB_MODE=disabled, skipping wandb tracker initialization.")
        return None

    if "wandb_project" in training_config:
        os.environ["WANDB_PROJECT"] = training_config["wandb_project"]
    if "wandb_name" in training_config:
        os.environ["WANDB_NAME"] = training_config["wandb_name"]
    if "wandb_entity" in training_config:
        os.environ["WANDB_ENTITY"] = training_config["wandb_entity"]

    import wandb

    kwargs = {
        "project": training_config.get("wandb_project", "TaH"),
        "config": config,
        "dir": os.path.join(output_dir, "wandb"),
    }
    if training_config.get("wandb_name"):
        kwargs["name"] = training_config["wandb_name"]
    if training_config.get("wandb_entity"):
        kwargs["entity"] = training_config["wandb_entity"]
    run = wandb.init(**kwargs)
    print(f"Wandb tracker initialized: project={kwargs['project']}")
    return run


def train_model_tp(
    model,
    tokenizer,
    processed_train_dataset,
    output_dir: str,
    training_config: Dict,
    data_config: Dict,
    run_config: Optional[Dict],
    mesh,
    shim,
    processed_eval_dataset=None,
    resume_from_checkpoint_path: Optional[str] = None,
) -> List[float]:
    """Torch-native FSDP2(dp) × DTensor-TP(tp) training loop (torchrun launch).

    ``model`` must already be sharded via ``apply_tensor_parallel`` +
    ``apply_fsdp``. Semantics match the accelerate-FSDP trainer: fp32 sharded
    masters with bf16 compute, loss normalized by the GLOBAL token count
    (``loss * dp`` compensates FSDP's mean reduction), fp32 per-microbatch
    grad accumulation. Eval is loss-only. Checkpoints always carry the model +
    progress; with ``save_only_model: false`` they also carry the sharded
    optimizer state (DCP), and resume restores whatever is present.
    """
    import time

    import torch.distributed as dist

    from tah2.train.dynamic_batch import BalancedGlobalBatchSampler, _materialize_lengths
    from tah2.train.tp import (
        clip_grad_norm_tp,
        grad_norm_tp,
        split_dtensor_param_groups,
        sync_replicated_grads,
    )

    device = shim.device
    rank = dist.get_rank()
    dp_size = mesh["dp"].size()
    tp_size = mesh["tp"].size()
    dp_group = mesh.get_group("dp") if dp_size > 1 else None
    tp_group = mesh.get_group("tp") if tp_size > 1 else None
    dp_rank = mesh.get_local_rank("dp")
    wandb_run = setup_wandb_tp(training_config, run_config or {}, output_dir, rank)

    dyn_cfg = training_config.get("dynamic_batch") or {}
    if not dyn_cfg.get("enabled", False) or dyn_cfg.get("global_batch_samples") is None:
        raise ValueError(
            "train_model_tp requires dynamic_batch with global_batch_samples."
        )
    gbs = int(dyn_cfg["global_batch_samples"])

    with shim.main_process_first():
        lengths = _materialize_lengths(
            processed_train_dataset,
            data_config=data_config,
            cache_root=os.path.join(output_dir, ".cache"),
        )
    sampler = BalancedGlobalBatchSampler(
        lengths=lengths,
        global_batch_size=gbs,
        token_budget=int(dyn_cfg.get("token_budget", 8192)),
        max_batch_size=dyn_cfg.get("max_batch_size", None),
        num_replicas=dp_size,
        seed=int(dyn_cfg.get("seed", 420)),
        drop_last=True,
    )
    collator = CustomTaHDataCollator(tokenizer=tokenizer, padding=True)

    num_train_epochs = int(math.ceil(float(training_config.get("num_train_epochs", 1))))
    initial_mb_counts = sampler.per_rank_mb_counts()
    windows_per_epoch = len(initial_mb_counts)
    max_steps = windows_per_epoch * num_train_epochs

    optimizer = build_optimizer(
        model, training_config, training_config.get("separate_lr", None), shim
    )
    # FSDP2 keeps params as fp32 sharded masters (bf16 is compute-only); split
    # groups so fused AdamW gets uniform mesh/placement operands.
    split_dtensor_param_groups(optimizer)
    lr_scheduler = create_scheduler(optimizer, training_config, max_steps)
    max_grad_norm = float(training_config.get("max_grad_norm", 0.0) or 0.0)
    logging_steps = max(int(training_config.get("logging_steps", 1)), 1)
    max_steps_override = int(training_config.get("max_steps", 0) or 0)
    save_strategy = str(training_config.get("save_strategy", "no")).lower()
    save_steps = int(training_config.get("save_steps", 0) or 0)
    save_total_limit = training_config.get("save_total_limit", None)
    save_only_model = bool(training_config.get("save_only_model", False))

    state = SimpleNamespace(global_step=0, epoch=0.0, max_steps=max_steps)
    metric_logger = MetricLogger(SimpleNamespace(accelerator=None, state=state))
    # Reduce iter/loss metrics over dp only (tp ranks share data).
    metric_logger.dp_group = dp_group
    model.logger_callback = metric_logger
    # Log iter_ge{k} bins up to the model's max_iter (rank-invariance is
    # documented at the all_reduce in MetricLogger.log_iter_metrics).
    _max_iter = getattr(getattr(model, "tah_config", None), "max_iter", None)
    if _max_iter:
        metric_logger.max_depth_bins = int(_max_iter)
    sync_training_state(model, state, num_train_epochs)

    # Optional depth curriculum: training.max_iter_schedule = [[start_step, max_iter], ...].
    # The last entry with start_step <= global_step wins, clamped to the model's
    # configured max_iter (decider/labeler were built for it). All ranks derive the
    # target from the same state.global_step, keeping per-iter collectives aligned.
    _mi_schedule = training_config.get("max_iter_schedule", None)
    _mi_cap = int(getattr(model, "max_iter", 0) or 0)
    if _mi_schedule:
        _mi_schedule = sorted((int(s), int(m)) for s, m in _mi_schedule)
        if rank == 0:
            print(
                f"max_iter_schedule enabled: {_mi_schedule} (cap {_mi_cap})",
                flush=True,
            )

    def _apply_max_iter_schedule(global_step: int) -> None:
        if not _mi_schedule:
            return
        target = _mi_schedule[0][1]
        for _s, _m in _mi_schedule:
            if global_step >= _s:
                target = _m
        target = max(1, min(target, _mi_cap or target))
        if int(model.max_iter) != target:
            if rank == 0:
                print(
                    f"[max_iter_schedule] step {global_step}: max_iter -> {target}",
                    flush=True,
                )
            model.max_iter = target

    _apply_max_iter_schedule(state.global_step)

    def _save_train_state(ckpt_dir: str, epoch_payload: int) -> None:
        """Progress (always) + sharded fp32 masters & optimizer state (unless
        save_only_model). The HF checkpoint is bf16; exact resume needs the
        fp32 masters, so they go into the DCP state alongside the optimizer."""
        if not save_only_model:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                get_optimizer_state_dict,
            )

            dcp.save(
                {
                    "model": get_model_state_dict(model),
                    "optim": get_optimizer_state_dict(model, optimizer),
                },
                checkpoint_id=os.path.join(ckpt_dir, "optim"),
            )
        if rank == 0:
            torch.save(
                {
                    "global_step": state.global_step,
                    "epoch": epoch_payload,
                    "windows_per_epoch": windows_per_epoch,
                    "dp": dp_size,
                    "tp": tp_size,
                },
                os.path.join(ckpt_dir, "simple_trainer_state.pt"),
            )

    # ---- resume: weights were already loaded from the checkpoint by the
    # caller (model.tah_model_path); restore progress + optimizer here.
    start_epoch = 0
    skip_windows = 0
    if resume_from_checkpoint_path:
        state_file = os.path.join(
            resume_from_checkpoint_path, "simple_trainer_state.pt"
        )
        if os.path.isfile(state_file):
            payload = torch.load(state_file, map_location="cpu")
            saved_wpe = int(payload.get("windows_per_epoch", windows_per_epoch))
            if saved_wpe != windows_per_epoch:
                raise ValueError(
                    f"resume: windows_per_epoch changed ({saved_wpe} -> "
                    f"{windows_per_epoch}); data/batching must match the saved run."
                )
            state.global_step = int(payload.get("global_step", 0))
            start_epoch = state.global_step // windows_per_epoch
            skip_windows = state.global_step % windows_per_epoch
            # Fast-forward a fresh scheduler instead of loading its state:
            # exact for step-count-based schedules and topology-independent.
            for _ in range(state.global_step):
                lr_scheduler.step()
            optim_dir = os.path.join(resume_from_checkpoint_path, "optim")
            if os.path.isdir(optim_dir):
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint.state_dict import (
                    get_model_state_dict,
                    get_optimizer_state_dict,
                    set_model_state_dict,
                    set_optimizer_state_dict,
                )

                # Restores the exact fp32 masters (the HF checkpoint the
                # weights were loaded from is bf16-rounded) + optimizer state.
                payload = {
                    "model": get_model_state_dict(model),
                    "optim": get_optimizer_state_dict(model, optimizer),
                }
                dcp.load(payload, checkpoint_id=optim_dir)
                set_model_state_dict(model, model_state_dict=payload["model"])
                set_optimizer_state_dict(
                    model, optimizer, optim_state_dict=payload["optim"]
                )
                shim.print(f"resume: fp32 masters + optimizer loaded from {optim_dir}")
            else:
                shim.print(
                    "resume: no optimizer state in checkpoint "
                    "(save_only_model) — fresh optimizer, bf16-rounded weights"
                )
            shim.print(
                f"resume: global_step={state.global_step} epoch={start_epoch} "
                f"skip_windows={skip_windows}"
            )
        else:
            shim.print(
                f"resume: no simple_trainer_state.pt under "
                f"{resume_from_checkpoint_path}; starting from step 0 (weights only)"
            )

    mb_counts = sorted(set(initial_mb_counts))
    mb_summary = (
        str(mb_counts[0])
        if len(mb_counts) == 1
        else f"{mb_counts[0]}-{mb_counts[-1]}"
    )
    shim.print(
        f"sampler: gbs={gbs} micro_batches/rank={mb_summary} "
        f"windows/epoch={windows_per_epoch} epochs={num_train_epochs} max_steps={max_steps}"
    )

    class _RankStridedBatches:
        """Lazy per-dp-rank view (index % world == rank) of the microbatch
        stream; the first ``skip`` global microbatches (already-trained
        windows on resume) are dropped without being collated."""

        def __init__(self, base, rank, world, skip=0):
            self.base, self.rank, self.world, self.skip = base, rank, world, skip

        def __iter__(self):
            for i, batch in enumerate(self.base):
                if i >= self.skip and i % self.world == self.rank:
                    yield batch

        def __len__(self):
            return sum(
                1
                for i in range(self.skip, len(self.base))
                if i % self.world == self.rank
            )

    num_workers = int(training_config.get("dataloader_num_workers", 4))

    # ---- eval (loss-only), summed loss all-reduced over dp ----
    eval_strategy = str(training_config.get("eval_strategy", "no")).lower()
    eval_steps = int(training_config.get("eval_steps", 0) or 0)
    eval_on_start = bool(training_config.get("eval_on_start", False))
    eval_loader = None
    _n_eval_batches = 0
    eval_num_items = eval_gbs = None
    if processed_eval_dataset is not None and eval_strategy != "no":
        eval_loader = DataLoader(
            processed_eval_dataset,
            batch_size=int(training_config.get("per_device_eval_batch_size", 1)),
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
            num_workers=num_workers,
        )
        eval_num_items = torch.tensor(
            float(compute_eval_num_items(processed_eval_dataset)), device=device
        )
        eval_gbs = torch.tensor(
            float(compute_eval_global_batch_size(processed_eval_dataset)),
            device=device,
        )
        # Equal per-dp-rank batch counts keep TaH's WORLD collectives aligned.
        _n_eval_batches = len(eval_loader) - (len(eval_loader) % dp_size)
        if _n_eval_batches != len(eval_loader):
            shim.print(
                f"eval: using {_n_eval_batches}/{len(eval_loader)} batches "
                f"(divisible by dp={dp_size})"
            )

    history: List[float] = []
    progress_bar = tqdm(
        total=max_steps,
        initial=min(state.global_step, max_steps),
        desc="Training",
        disable=rank != 0,
    )

    def _run_eval():
        model.eval()
        loss_sum = torch.tensor(0.0, device=device)
        eval_progress = None
        if rank == 0:
            eval_progress = tqdm(
                total=_n_eval_batches // dp_size,
                desc="Eval",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
        try:
            with torch.no_grad():
                for i, batch in enumerate(
                    itertools.islice(eval_loader, _n_eval_batches)
                ):
                    if i % dp_size != dp_rank:
                        continue
                    batch = {
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(
                            **batch,
                            num_items_in_batch=eval_num_items,
                            global_batch_size=eval_gbs,
                            logger_callback=metric_logger,
                        )
                    loss_sum += out.loss.detach().float()
                    if eval_progress is not None:
                        eval_progress.update(1)
        finally:
            if eval_progress is not None:
                eval_progress.close()
        if dp_group is not None:
            dist.all_reduce(loss_sum, group=dp_group)
        metrics = {"eval_loss": float(loss_sum.item())}
        metrics.update(metric_logger.pop_eval_logs())
        model.train()
        eval_logs = _prefix_metrics(_format_metrics(metrics, state.global_step), "eval")
        if wandb_run is not None:
            wandb_run.log(eval_logs, step=state.global_step)
        if rank == 0:
            # keep the eval/ prefix so the line is distinguishable from train logs
            progress_bar.write(str(eval_logs))

    if eval_loader is not None and eval_on_start:
        _run_eval()
    for epoch in range(start_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        per_rank_counts = sampler.per_rank_mb_counts()
        # On the resumed epoch, drop the already-trained windows' microbatches
        # at the sampler level (never collated).
        epoch_skip = skip_windows if epoch == start_epoch else 0
        skip_mbs = sum(per_rank_counts[:epoch_skip]) * dp_size
        # tp ranks in a dp group consume the same stream -> routing is
        # tp-rank-consistent.
        loader = DataLoader(
            processed_train_dataset,
            batch_sampler=_RankStridedBatches(sampler, dp_rank, dp_size, skip_mbs),
            collate_fn=collator,
            pin_memory=True,
            num_workers=num_workers,
        )
        mb_iter = iter(loader)

        state.epoch = float(epoch)
        sync_training_state(model, state, num_train_epochs)

        for window_idx, gas_now in enumerate(per_rank_counts):
            if window_idx < epoch_skip:
                continue
            _apply_max_iter_schedule(state.global_step)
            step_start = time.perf_counter()
            batch_samples = list(itertools.islice(mb_iter, gas_now))
            if not batch_samples:
                break
            batch_samples = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                for b in batch_samples
            ]

            num_items_local = sum(
                int((b["labels"] != -100).sum()) for b in batch_samples
            )
            batch_size_local = sum(int(b["labels"].shape[0]) for b in batch_samples)
            num_items = torch.tensor(float(num_items_local), device=device)
            global_batch_size = torch.tensor(float(batch_size_local), device=device)
            # dp-only reduction: tp ranks share the same data, do not recount.
            if dp_group is not None:
                dist.all_reduce(num_items, group=dp_group)
                dist.all_reduce(global_batch_size, group=dp_group)

            step_loss = torch.tensor(0.0, device=device)
            for batch in batch_samples:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        **batch,
                        num_items_in_batch=num_items,
                        global_batch_size=global_batch_size,
                        load_balance_num_micro_batches=len(batch_samples),
                        logger_callback=metric_logger,
                    )
                # Global-token-normalized loss needs a dp SUM of gradients;
                # FSDP's reduce averages, hence the dp factor.
                (outputs.loss * dp_size).backward()
                step_loss += outputs.loss.detach().float()

            sync_replicated_grads(model, tp_group)
            # Per-module pre-clip grad norms (TaH side modules), so a spike can be
            # attributed to backbone / updater / decider. Collectives: all ranks.
            grad_norm_updater = grad_norm_tp(model.input_updater.parameters())
            grad_norm_decider = grad_norm_tp(model.iter_decider.parameters())
            # PerDepthIterDecider: one norm per depth so a dead/exploding depth shows.
            grad_norm_decider_by_depth = {
                d + 1: grad_norm_tp(sub.parameters())
                for d, sub in enumerate(getattr(model.iter_decider, "deciders", []))
            }
            grad_norm = clip_grad_norm_tp(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state.global_step += 1
            state.epoch = epoch + (window_idx + 1) / max(windows_per_epoch, 1)
            sync_training_state(model, state, num_train_epochs)

            if dp_group is not None:
                dist.all_reduce(step_loss, group=dp_group)
            if state.global_step % logging_steps == 0 and rank == 0:
                torch.cuda.synchronize()
                logs = {
                    "step": state.global_step,
                    "loss": round(step_loss.item(), 4),
                    # .4g matches the accelerate path's _format_metrics
                    "learning_rate": float(f"{lr_scheduler.get_last_lr()[0]:.4g}"),
                    "num_items_in_batch": float(num_items.item()),
                    "global_batch_size": float(global_batch_size.item()),
                    "grad_norm": round(float(grad_norm.item()), 4),
                    "grad_norm_updater": round(float(grad_norm_updater.item()), 4),
                    "grad_norm_decider": round(float(grad_norm_decider.item()), 4),
                    **{
                        f"grad_norm_decider_d{d}": round(float(g.item()), 4)
                        for d, g in grad_norm_decider_by_depth.items()
                    },
                    "step_time_s": round(time.perf_counter() - step_start, 2),
                    "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
                extra = metric_logger.pop_train_logs(step_count=1)
                logs.update({k: round(float(v), 4) for k, v in extra.items()})
                train_logs = _prefix_metrics(
                    _format_metrics(logs, state.global_step), "train"
                )
                if wandb_run is not None:
                    wandb_run.log(train_logs, step=state.global_step)
                progress_bar.write(str(_terminal_metrics(train_logs)))
            else:
                metric_logger.pop_train_logs(step_count=1)
            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=round(float(step_loss.item()), 4),
                    lr=float(f"{lr_scheduler.get_last_lr()[0]:.4g}"),
                )
            history.append(float(step_loss.item()))

            if (
                eval_loader is not None
                and eval_strategy == "steps"
                and eval_steps > 0
                and state.global_step % eval_steps == 0
            ):
                _run_eval()

            if (
                save_strategy == "steps"
                and save_steps > 0
                and state.global_step % save_steps == 0
            ):
                ckpt_dir = os.path.join(
                    output_dir, f"checkpoint-{state.global_step}"
                )
                save_model_tp(model, tokenizer, ckpt_dir, shim)
                _save_train_state(ckpt_dir, epoch)
                if rank == 0:
                    rotate_checkpoints(output_dir, save_total_limit, shim)
                dist.barrier()

            if max_steps_override and state.global_step >= max_steps_override:
                break

        if max_steps_override and state.global_step >= max_steps_override:
            shim.print(f"early-exit: reached max_steps={max_steps_override}")
            break

        if eval_loader is not None and eval_strategy == "epoch":
            _run_eval()

        if save_strategy == "epoch":
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{state.global_step}")
            save_model_tp(model, tokenizer, ckpt_dir, shim)
            _save_train_state(ckpt_dir, epoch + 1)
            if rank == 0:
                rotate_checkpoints(output_dir, save_total_limit, shim)
            dist.barrier()

    progress_bar.close()
    if rank == 0 and history:
        print(
            f"--- done. steps={state.global_step} final_loss={history[-1]:.6f} ---",
            flush=True,
        )
    if wandb_run is not None:
        wandb_run.finish()
    return history
