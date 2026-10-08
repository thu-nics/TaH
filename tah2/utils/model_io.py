from __future__ import annotations

from typing import Optional

from tah2.utils.component_io import (
    freeze_parameters_when_flag_false,
    load_registered_component,
    save_registered_component,
)
from tah2.utils.component_registry import get_input_updater_class, get_iter_decider_class


def save_input_updater(updater, save_directory: str, state_dict=None) -> None:
    save_registered_component(
        updater, save_directory, "input_updater.bin", state_dict=state_dict
    )


def load_input_updater(
    load_directory: str,
    class_name: Optional[str] = None,
    init_args: Optional[dict] = None,
):
    return load_registered_component(
        load_directory,
        get_input_updater_class,
        class_name=class_name,
        init_args=init_args,
        load_name="input_updater.bin",
        post_load=freeze_parameters_when_flag_false,
    )


def save_iter_decider(
    iter_decider,
    save_directory: str,
    save_name: Optional[str] = None,
    state_dict=None,
) -> None:
    save_registered_component(
        iter_decider,
        save_directory,
        save_name if save_name is not None else "iter_decider.bin",
        path_keys=("decider_config_path",),
        state_dict=state_dict,
    )


def load_iter_decider(
    load_directory: str,
    class_name: Optional[str] = None,
    init_args: Optional[dict] = None,
    load_name: Optional[str] = None,
):
    return load_registered_component(
        load_directory,
        get_iter_decider_class,
        class_name=class_name,
        init_args=init_args,
        load_name=load_name if load_name is not None else "iter_decider.bin",
        path_keys=("decider_config_path",),
    )
