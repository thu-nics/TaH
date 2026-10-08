#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import yaml


def build_output_dir(config_path: str, run_ts: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_config = cfg["model"]
    data_config = cfg["data"]

    output_dir = data_config["output_dir"]
    if "tah_model_path" in model_config:
        base_model_name = model_config["name"].split("/")[-1]
        output_dir = os.path.join(
            output_dir, "continue_training", base_model_name, run_ts
        )
    else:
        # Component names distinguish training runs using the core recipe.
        updater = model_config["input_updater"]
        decider = model_config["iter_decider"]
        output_dir = os.path.join(
            output_dir,
            model_config["name"].split("/")[-1] + "_" + updater[: -len("Updater")],
        )
        output_dir = output_dir + "_" + decider[: -len("IterDecider")]
        output_dir = os.path.join(output_dir, run_ts)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and create the training output directory from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the training config YAML."
    )
    parser.add_argument(
        "--run-ts", required=True, help="Timestamp suffix for this run."
    )
    args = parser.parse_args()

    output_dir = build_output_dir(args.config, args.run_ts)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Verbatim copy of the recipe, comments and key order intact. SFT_TaH.py also
    # dumps the *parsed* config as training_config.yaml (what the analysis scripts
    # read); this one is the source of truth for what was actually launched.
    shutil.copyfile(args.config, os.path.join(output_dir, "config.yaml"))
    print(output_dir)


if __name__ == "__main__":
    main()
