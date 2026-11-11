"""Capture compact per-position records for one implementation (R1-O1).

    python -m script.tah.validation.capture reference --dtype fp32 --layout chunk --mode free ...   # oracle
    python -m script.tah.validation.capture reference --dtype bf16 --layout generate --batch 8 \
        --mode forced --target <oracle run> ...
    python -m script.tah.validation.capture engine --mode forced --target <oracle run> ...

Forced mode imposes the target's decisions and measures full-vocabulary logit errors
against the target: the implementation's native-dtype logits at generated positions
are dumped to a transient memmap, then the target (oracle) replays each sequence
in its original free mode and reduces |x - o| per position. When a block's dump exceeds
the buffer budget the block is replayed once per subset; every replay must reproduce
the first one bitwise (checked by fingerprint). Completed blocks are skipped on rerun.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .common import (DTYPES, LogitDump, dump_subsets, effective_threshold, fingerprint, load_manifest, load_record,
                     load_reference, model_sha256, record_path, save_record, select, write_json)
from .fp64_attention import POLICY as FP64_ATTENTION_POLICY
from .reference_run import ReferenceRunner


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("impl", choices=["reference", "engine"])
    p.add_argument("--manifest", type=Path, required=True, help="sequence manifest (select_sequences.py)")
    p.add_argument("--sequence-order", choices=["manifest", "reverse"], default="manifest",
                   help="validation composition diagnostic: reverse selected sequence order")
    p.add_argument("--subset", default="all", help="'all', a manifest subset name, or first:N")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, default=None, help="TaH config override (real checkpoint)")
    p.add_argument("--out", type=Path, required=True, help="run directory (records/ + run.json)")
    p.add_argument("--mode", choices=["free", "forced"], required=True)
    p.add_argument("--target", type=Path, default=None,
                   help="oracle run directory: decisions to force and logits to compare against")
    p.add_argument("--no-compare", action="store_true", help="forced mode without logit comparison")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bf16", help="reference precision")
    p.add_argument("--layout", choices=["generate", "chunk"], default="generate", help="reference layout")
    p.add_argument("--batch", type=int, default=8, help="reference generate-layout batch size")
    p.add_argument("--chunk", type=int, default=256, help="reference chunk-layout width")
    p.add_argument("--block", type=int, default=100, help="engine: sequences per generate() call")
    p.add_argument("--max-batch-size", type=int, default=32, help="engine resident rows")
    p.add_argument("--prefill-chunk-size", type=int, default=128)
    p.add_argument("--sampling-topk", type=int, default=128)
    p.add_argument("--memory-fraction", type=float, default=.9)
    p.add_argument("--kv-capacity", type=int, default=None, help="engine KV entries (default: measured once)")
    p.add_argument("--host-swap-gb", type=float, default=8.,
                   help="engine host swap arena, pinned host RAM held while each engine replay runs (one at a time)")
    p.add_argument("--buffer-dir", type=Path, default=Path("data/tmp/oracle-buffer"))
    p.add_argument("--buffer-gb", type=float, default=16., help="logit dump budget per replay")
    p.add_argument("--keep-target", action="store_true",
                   help="keep the target model on the device for the whole run (faster; needs memory for "
                        "both models). Default: load it per comparison and park the reference model on the CPU")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)
    if args.mode == "forced" and args.target is None:
        p.error("--mode forced needs --target")
    if args.impl == "reference" and args.layout == "chunk" and args.batch != 8:
        p.error("--batch applies to the generate layout only")
    return args


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def environment():
    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance only
            return None
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")))


class Target:
    """The oracle run being compared against, rebuilt from its run.json."""

    def __init__(self, directory, device):
        self.dir = Path(directory)
        self.info = json.loads((self.dir / "run.json").read_text())
        cfg = self.info["config"]
        if cfg["impl"] != "reference" or cfg["mode"] != "free" or cfg["layout"] != "chunk":
            raise ValueError("a target must be a free-mode reference run in the chunk layout")
        if cfg.get("dtype") == "fp64" and cfg.get("fp64_attention") != FP64_ATTENTION_POLICY:
            raise ValueError(f"{self.dir}: float64 oracle captured without the PyTorch #143644 attention "
                             f"workaround ({FP64_ATTENTION_POLICY}); its attention may be wrong, recapture it")
        self.device = device
        self.model = None

    def decisions(self, seq):
        record = load_record(self.dir, seq["id"])
        return record["decision"]

    def load(self):
        if self.model is None:
            cfg = self.info["config"]
            self.model = ReferenceRunner(load_reference(cfg["model"], cfg["eval_config"], cfg["dtype"], self.device))
        return self.model

    def unload(self):
        self.model = None
        free_memory()

    def compare(self, seqs, dump, arrays):
        """Replay the original free oracle exactly; check identity and fill sq/maxabs.

        Forcing recorded decisions is a different execution path: it allocates a
        replacement device decision tensor in every hook. Even equal decisions
        do not establish identical floating-point execution. The implementations
        being measured still have the recorded oracle decisions imposed.
        """
        cfg = self.info["config"]
        runner = self.load()
        for seq in seqs:
            reference = load_record(self.dir, seq["id"])
            sq, maxabs = arrays[seq["id"]]["sq"], arrays[seq["id"]]["maxabs"]

            def consume(s, indices, logits):
                start, end = indices[0], indices[-1] + 1
                if indices != list(range(start, end)):
                    raise AssertionError("target chunk positions are not contiguous")
                x = dump.read(s["id"], start, end, logits.device).double()
                d = x - logits.double()
                sq[start:end] = d.square().sum(-1).cpu().numpy()
                maxabs[start:end] = d.abs().amax(-1).cpu().numpy()
            rec = runner.run([seq], layout="chunk", chunk=cfg["chunk"], pad_id=0,
                             consumer=consume)[0].arrays()
            differences = replay_differences(reference, rec)
            if differences:
                raise AssertionError(f"{seq['id']}: target recomputation does not reproduce its record "
                                     f"(strict oracle replay failed): {json.dumps(differences, sort_keys=True)}")


def replay_differences(reference, replay):
    """Compact exact mismatch diagnostics; no numerical acceptance tolerance."""
    differences = {}
    for key in ("visited", "decision", "prob", "top1", "gap"):
        expected, actual = reference[key], replay[key]
        if expected.shape != actual.shape:
            differences[key] = dict(expected_shape=list(expected.shape), actual_shape=list(actual.shape))
            continue
        same = expected == actual
        if key in ("prob", "gap"):
            same |= np.isnan(expected) & np.isnan(actual)
        if same.all():
            continue
        where = ~same
        first = tuple(np.argwhere(where)[0])
        row = dict(count=int(where.sum()), first_index=[int(i) for i in first],
                   expected=float(expected[first]), actual=float(actual[first]))
        if key in ("prob", "gap"):
            finite = where & np.isfinite(expected) & np.isfinite(actual)
            row["max_abs_finite_difference"] = float(np.abs(expected[finite]-actual[finite]).max()) if finite.any() else None
            row["nonfinite_mismatches"] = int((where & ~finite).sum())
        differences[key] = row
    return differences


def blocks_of(args, seqs):
    size = args.block if args.impl == "engine" else (args.batch if args.layout == "generate" else 1)
    return [seqs[i:i + size] for i in range(0, len(seqs), size)]


class Implementation:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.reference = None
        self.kv_capacity = args.kv_capacity
        self.dtype = DTYPES[args.dtype] if args.impl == "reference" else torch.bfloat16
        self.threshold = None

    def park(self):
        if self.reference is not None:
            self.reference.model.to("cpu")
            free_memory()

    def unpark(self):
        if self.reference is not None:
            self.reference.model.to(self.args.device)

    def run(self, seqs, forced, consumer):
        args = self.args
        if args.impl == "reference":
            if self.reference is None:
                self.reference = ReferenceRunner(load_reference(args.model, args.eval_config, args.dtype, args.device))
            self.threshold = self.reference.threshold
            recorders = self.reference.run(seqs, layout=args.layout, chunk=args.chunk,
                                           pad_id=self.manifest.get("pad_token_id", 0), forced=forced,
                                           consumer=consumer)
            return recorders, {}
        import yaml
        from .engine_run import EngineRunner
        config = yaml.safe_load(args.eval_config.read_text()) if args.eval_config else None
        eos = self.manifest.get("eos_token_id", "tokenizer")
        runner = EngineRunner(args.model, config, device=args.device, max_batch_size=args.max_batch_size,
                              prefill_chunk_size=args.prefill_chunk_size, kv_capacity=self.kv_capacity,
                              memory_fraction=args.memory_fraction,
                              host_swap_bytes=int(args.host_swap_gb * 1024**3),
                              sampling_topk=args.sampling_topk, eos_token_id=eos)
        self.kv_capacity = runner.kv_capacity  # identical capacity on every replay
        self.threshold = effective_threshold(runner.engine.decider)
        try:
            return runner.run(seqs, forced=forced, consumer=consumer)
        finally:
            runner.close()
            del runner
            free_memory()


def main(argv=None):
    args = parse(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest)
    seqs = select(manifest, args.subset)
    if args.sequence_order == "reverse":
        seqs = list(reversed(seqs))
    args.out.mkdir(parents=True, exist_ok=True)
    compare = args.mode == "forced" and not args.no_compare
    target = Target(args.target, args.device) if args.target else None
    config = dict(impl=args.impl, mode=args.mode, dtype=args.dtype if args.impl == "reference" else "bf16",
                  layout=args.layout if args.impl == "reference" else "engine",
                  batch=args.batch, chunk=args.chunk, block=args.block, max_batch_size=args.max_batch_size,
                  prefill_chunk_size=args.prefill_chunk_size, sampling_topk=args.sampling_topk,
                  model=str(args.model), eval_config=str(args.eval_config) if args.eval_config else None,
                  manifest=str(args.manifest), subset=args.subset,
                  target=str(args.target) if args.target else None, compare=compare,
                  oracle_constants="bf16-rounded buffers including inv_freq (R1-O amendment 2026-10-13)",
                  eval_config_sha256=__import__("hashlib").sha256(args.eval_config.read_bytes()).hexdigest()
                  if args.eval_config else None)
    if args.sequence_order != "manifest":
        config["sequence_order"] = args.sequence_order
    if config["dtype"] == "fp64":
        config["fp64_attention"] = FP64_ATTENTION_POLICY  # ReferenceRunner applies it to every fp64 run
    # --keep-target only changes memory placement, not results; it is not part of the configuration.
    run_path = args.out / "run.json"
    info = json.loads(run_path.read_text()) if run_path.exists() else None
    if info is not None and info["config"] != config:
        raise SystemExit(f"{run_path} was written with a different configuration; use a new --out")
    if info is None:
        info = dict(config=config, model_sha256=model_sha256(args.model), environment=environment(),
                    manifest_sha256=__import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(),
                    blocks=[], sequences=[s["id"] for s in seqs])
        write_json(run_path, info)
    impl = Implementation(args, manifest)
    if compare and args.keep_target:
        # Resident for the whole run, so engine KV sizing already accounts for it.
        target.load()
    if args.kv_capacity is None and info["blocks"]:
        impl.kv_capacity = info["blocks"][0].get("stats", {}).get("kv_capacity")
    vocab = None
    for number, block in enumerate(blocks_of(args, seqs)):
        if all(record_path(args.out, s["id"]).exists() for s in block):
            continue
        started = time.perf_counter()
        forced = [target.decisions(s) for s in block] if args.mode == "forced" else None
        if vocab is None:
            vocab = json.loads((args.model / "config.json").read_text())["vocab_size"]
        itemsize = torch.empty((), dtype=impl.dtype).element_size()
        subsets = dump_subsets(block, vocab, itemsize, args.buffer_gb * 1024**3) if compare else [None]
        arrays, prints, stats = None, None, None
        for subset in subsets:
            dump = LogitDump(args.buffer_dir, subset, vocab, impl.dtype) if subset else None
            wanted = {s["id"] for s in subset} if subset else set()

            def consume(seq, indices, logits):
                if seq["id"] in wanted:
                    dump.write(seq["id"], indices, logits)
            try:
                recorders, run_stats = impl.run(block, forced, consume if dump else None)
                current = {r.seq["id"]: r.arrays() for r in recorders}
                current_prints = {k: fingerprint(v) for k, v in current.items()}
                if arrays is None:
                    arrays, prints, stats = current, current_prints, run_stats
                elif current_prints != prints:
                    raise AssertionError(f"block {number}: replay is not bitwise identical to the first run")
                if dump:
                    if not args.keep_target:
                        impl.park()
                    target.compare(subset, dump, arrays)
                    if not args.keep_target:
                        target.unload()
                        impl.unpark()
            finally:
                if dump:
                    dump.close()
            free_memory()
        for s in block:
            save_record(args.out, s["id"], arrays[s["id"]])
        info["threshold"] = impl.threshold
        info["blocks"].append(dict(index=number, sequences=[s["id"] for s in block], replays=len(subsets),
                                   seconds=time.perf_counter() - started, stats=stats,
                                   fingerprints=prints))
        write_json(run_path, info)
        print(json.dumps(dict(block=number, sequences=len(block), replays=len(subsets),
                              seconds=round(time.perf_counter() - started, 1))), flush=True)
    info["complete"] = all(record_path(args.out, s["id"]).exists() for s in seqs)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        info["peak_memory_bytes"] = max(info.get("peak_memory_bytes", 0), torch.cuda.max_memory_allocated())
    write_json(run_path, info)


if __name__ == "__main__":
    main()
