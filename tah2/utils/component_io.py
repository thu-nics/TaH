from __future__ import annotations

import os
import shutil
from typing import Callable, Iterable, Optional

import torch


def save_registered_component(
    component,
    save_directory: str,
    save_name: str,
    path_keys: Iterable[str] = (),
    state_dict=None,
) -> None:
    init_args = getattr(component, "_init_args", {}).copy()
    for key in path_keys:
        path = init_args.get(key)
        if path and os.path.exists(path):
            filename = os.path.basename(path)
            target = os.path.join(save_directory, filename)
            shutil.copy2(path, target)
            init_args[key] = filename
            print(f"Copied {key} to {target}")

    if state_dict is None:
        state_dict = component.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    # A sharded (DTensor) param here would silently save one rank's shard;
    # callers must gather full tensors first (see save_model_tp).
    for key, value in state_dict.items():
        if hasattr(value, "to_local"):
            raise RuntimeError(
                f"{save_name}: '{key}' is a DTensor shard; pass a gathered "
                "plain state_dict when saving sharded models."
            )
    save_path = os.path.join(save_directory, save_name)
    # print(f"Saving {save_name} with {len(state_dict)} parameters to {save_path}")
    torch.save(
        {
            "class": component.__class__.__name__,
            "state_dict": state_dict,
            "init_args": init_args,
        },
        save_path,
    )


def load_registered_component(
    load_directory: str,
    get_component_class: Callable[[str], type],
    *,
    class_name: Optional[str] = None,
    init_args: Optional[dict] = None,
    load_name: str,
    path_keys: Iterable[str] = (),
    post_load: Optional[Callable[[object, dict], None]] = None,
):
    path = os.path.join(load_directory, load_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No {load_name} found at {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    class_name = class_name or data.get("class")
    if not class_name:
        raise ValueError(f"No {load_name} class specified in saved data")

    init_args = dict(init_args or data.get("init_args", {}))
    for key in path_keys:
        value = init_args.get(key)
        if value and not os.path.isabs(value):
            init_args[key] = os.path.join(load_directory, value)
            print(f"Resolved {key} to {init_args[key]}")

    component = get_component_class(class_name)(**init_args)

    state_dict = data.get("state_dict", {})
    if state_dict:
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in init_args}
        skipped = len(state_dict) - len(filtered_state_dict)
        if skipped:
            for key in state_dict:
                if key in init_args:
                    print(f"Skipping state_dict key '{key}' as it conflicts with init_args")
        print(
            f"Loading {load_name} state dict with {len(filtered_state_dict)} parameters "
            f"(filtered from {len(state_dict)})"
        )
        if filtered_state_dict:
            component.load_state_dict(filtered_state_dict, strict=False)

    if post_load is not None:
        post_load(component, init_args)
    return component


def freeze_parameters_when_flag_false(component, init_args: dict, flag_name: str = "learnable_weight") -> None:
    value = init_args.get(flag_name, None)
    if isinstance(value, bool):
        should_freeze = value is False
    elif isinstance(value, str):
        should_freeze = value.strip().lower() in ("false", "0", "no")
    else:
        should_freeze = False
    if should_freeze:
        for param in component.parameters():
            param.requires_grad = False
