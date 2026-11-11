"""CPU-readable reference parity fixtures (no HF/PEFT model imports).

Each teacher-forced record contains unpadded positions, depth-major traces,
and a small, uniformly spaced subset of the raw decider hidden inputs.
``visited`` distinguishes unexecuted depths from a stop decision. Hidden
layer indices are HF output_hidden_states indices, including embedding at 0.
Full-vocabulary logit errors are measured during capture, not inferred from
the saved top-k. Generation pass counts follow SPEC §6 (sampling position).
"""
from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file


SCHEMA_VERSION = 1


class FixtureSuite:
    """Load a manifest lazily, then individual CPU tensor records as needed."""

    def __init__(self, root: str | Path, *, allow_partial: bool = False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported fixture schema")
        if not allow_partial and self.manifest.get("status") != "complete":
            raise ValueError("fixture suite is incomplete; finish capture before using parity gates")

    def records(self, kind: str | None = None, checkpoint: str | None = None):
        for record in self.manifest["records"]:
            if kind is not None and record["kind"] != kind:
                continue
            if checkpoint is not None and record["checkpoint"] != checkpoint:
                continue
            yield record

    def load(self, record: dict):
        path = (self.root / record["file"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("fixture path escapes suite")
        tensors = load_file(str(path), device="cpu")
        if record["kind"] == "teacher_forced":
            length = tensors["input_ids"].numel()
            if tensors["visited"].shape != tensors["decision"].shape:
                raise ValueError("invalid decision trace shape")
            if tensors["visited"].shape[1] != length:
                raise ValueError("invalid trace length")
            if not (tensors["pass_count"] == tensors["visited"].sum(0)).all():
                raise ValueError("pass counts disagree with visited depths")
            if tensors["final_topk_values"].shape != tensors["final_topk_indices"].shape:
                raise ValueError("invalid top-k shape")
            if tensors["final_logits_sample"].shape[0] != tensors["logit_positions"].numel():
                raise ValueError("invalid dense logit sample")
        elif record["kind"] == "greedy":
            if tensors["generated_ids"].shape != tensors["generation_pass_count"].shape:
                raise ValueError("invalid generated-token pass counts")
            if tensors["prompt_ids"].shape != tensors["prompt_pass_count"].shape:
                raise ValueError("invalid prompt pass counts")
        return tensors
