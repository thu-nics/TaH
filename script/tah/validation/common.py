"""Shared pieces: sequences, model loading, sampler distributions, compact records, logit dumps.

Position convention for a sequence with prompt P (length p) and forced continuation F
(the generated tokens, plus EOS when generation ended with EOS): the teacher-forced
input is S = P + F[:-1]; positions p-1 .. len(S)-1 are the *generated positions*
(position p-1+j sampled F[j]). Logit-based fields exist only at generated positions;
per-pass decider fields cover every position.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

T, TOP_K, TOP_P = 0.6, 20, 0.95
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
RECORD_VERSION = 1


# ---------------------------------------------------------------- sequences

def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    for seq in manifest["sequences"]:
        if not seq["prompt"] or not seq["forced"]:
            raise ValueError(f"sequence {seq['id']} needs a prompt and a continuation")
    return manifest


def select(manifest, subset):
    sequences = manifest["sequences"]
    if subset in (None, "all"):
        return sequences
    if subset in manifest.get("subsets", {}):
        wanted = set(manifest["subsets"][subset])
        return [s for s in sequences if s["id"] in wanted]
    if subset.startswith("first:"):
        return sequences[:int(subset.split(":", 1)[1])]
    raise ValueError(f"unknown subset {subset!r}")


def inputs_of(seq):
    """Teacher-forced input ids and the first generated position."""
    return list(seq["prompt"]) + list(seq["forced"][:-1]), len(seq["prompt"]) - 1


# ---------------------------------------------------------------- models

def set_exact_float32():
    """Oracle arithmetic: no TF32 anywhere."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_reference(model_path, eval_config, dtype, device):
    """Unmodified reference model, weights as the bf16 implementations use them.

    Loads bf16 weights (as every bf16 run does), then casts exactly to ``dtype``;
    the oracle thus differs from the reference only in arithmetic precision. The
    decider threshold is the checkpoint's effective bf16 value (a model constant).
    """
    from dataclasses import fields
    import yaml
    from tah.model.tah_config import TaHConfig
    from tah.model.tah_model import TaHForCausalLM
    override = None
    if eval_config:
        raw = yaml.safe_load(Path(eval_config).read_text())
        valid = {f.name for f in fields(TaHConfig)}
        override = TaHConfig(**{k: v for k, v in raw.items() if k in valid})
    model = TaHForCausalLM.from_pretrained(str(model_path), tah_config=override,
                                           torch_dtype=torch.bfloat16, device_map="cpu",
                                           attn_implementation="sdpa")
    target = DTYPES[dtype] if isinstance(dtype, str) else dtype
    # Evaluation casts ALL buffers to bf16, including RoPE inv_freq. Round these
    # model constants before casting up: R1-O amendment (2026-10-13).
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    decider = model._active_iter_decider
    threshold = effective_threshold(decider)
    model = model.to(device=device, dtype=target).eval()
    decider = model._active_iter_decider
    if isinstance(decider.threshold, torch.Tensor):
        with torch.no_grad():
            decider.threshold.fill_(threshold)
        if float(decider.threshold) != threshold:
            raise AssertionError("decider threshold changed while casting")
    if target != torch.bfloat16:
        set_exact_float32()
    return model


def effective_threshold(decider):
    value = float(decider.threshold) if isinstance(decider.threshold, torch.Tensor) else float(decider.threshold)
    return float(torch.tensor(value, dtype=torch.bfloat16))


def model_sha256(path):
    """Hash of the weight/decider files (identifies the checkpoint in run.json)."""
    digest = hashlib.sha256()
    root = Path(path)
    for file in sorted(list(root.glob("*.safetensors")) + list(root.glob("iter_decider.bin")) +
                       list(root.glob("lora/*.safetensors")) + list(root.glob("tah_config.json"))):
        digest.update(file.name.encode())
        with open(file, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- sampling distributions

def reference_distribution(logits):
    """Reference ``sample_next_token`` math (multinomial intercepted), sparse rows.

    Runs in the logits' dtype, as the reference does; the intercepted weights are
    renormalized (multinomial normalizes them) and returned as float64 CSR pieces.
    """
    from tah.model.utils import sample_next_token
    captured = {}
    original = torch.multinomial

    def intercept(p, num_samples, *args, **kwargs):
        captured["p"] = p
        return torch.zeros(p.shape[0], num_samples, dtype=torch.long, device=p.device)
    torch.multinomial = intercept
    try:
        sample_next_token(logits, temperature=T, top_p=TOP_P, top_k=TOP_K, min_p=0., do_sample=True)
    finally:
        torch.multinomial = original
    return sparse(captured["p"])


class EngineDistribution:
    """Production engine sampler math (GraphSampler bucket the engine would use)."""

    def __init__(self, vocab_size, sampling_topk):
        from tah.engine.sampling import GraphSampler
        self.sampler = GraphSampler(vocab_size, max_top_k=sampling_topk)
        self.batches = {}

    def __call__(self, logits):
        from tah.engine.sampling import SamplingParams
        n = logits.shape[0]
        if n not in self.batches:
            self.batches[n] = self.sampler.make_batch([SamplingParams(temperature=T, top_k=TOP_K, top_p=TOP_P)] * n,
                                                      device=logits.device)
        ids, probs = self.sampler._distribution(logits, self.batches[n])
        dense = torch.zeros(logits.shape, dtype=torch.float64, device=logits.device)
        dense.scatter_(-1, ids, probs.double())
        return sparse(dense)


def sparse(p):
    """[n, V] weights -> per-row (ids int32, probs float64 normalized)."""
    p = p.double()
    p = p / p.sum(-1, keepdim=True)
    rows = []
    nz = p > 0
    counts = nz.sum(-1).tolist()
    ids = nz.nonzero()[:, 1].to(torch.int32).cpu().numpy()
    values = p[nz].cpu().numpy()
    start = 0
    for count in counts:
        rows.append((ids[start:start + count], values[start:start + count]))
        start += count
    return rows


def top_gap(logits):
    """Top-1 id and top-1 minus top-2 logit (float64) per row."""
    values, ids = logits.topk(2, dim=-1)
    return ids[:, 0].to(torch.int32).cpu().numpy(), (values[:, 0].double() - values[:, 1].double()).cpu().numpy()


# ---------------------------------------------------------------- compact records

class Recorder:
    """Per-sequence compact record being filled during one run."""

    def __init__(self, seq, max_iter):
        ids, first = inputs_of(seq)
        self.seq, self.length, self.first, self.max_iter = seq, len(ids), first, max_iter
        n = self.length - first
        self.visited = np.zeros((max_iter, self.length), bool)
        self.decision = np.zeros((max_iter, self.length), bool)
        self.own_decision = np.zeros((max_iter, self.length), bool)
        self.prob = np.full((max_iter, self.length), np.nan)
        self.top1 = np.full(n, -1, np.int32)
        self.gap = np.full(n, np.nan)
        self.dist = [None] * n
        self.sq = np.full(n, np.nan)
        self.maxabs = np.full(n, np.nan)

    def decide(self, depth, position, decision, own, prob):
        d = depth - 1
        self.visited[d, position] = True
        self.decision[d, position] = decision
        self.own_decision[d, position] = own
        # Pass-cap decisions carry no learned probability.
        self.prob[d, position] = prob if depth < self.max_iter else np.nan

    def final(self, positions, top1, gap, dist):
        for k, position in enumerate(positions):
            j = position - self.first
            self.top1[j], self.gap[j], self.dist[j] = top1[k], gap[k], dist[k]

    def arrays(self):
        if (self.top1 < 0).any() or any(d is None for d in self.dist):
            raise AssertionError(f"{self.seq['id']}: missing final logits at a generated position")
        if not self.visited[0].all():
            raise AssertionError(f"{self.seq['id']}: position without a first pass")
        counts = [len(d[0]) for d in self.dist]
        ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return dict(visited=self.visited, decision=self.decision, own_decision=self.own_decision,
                    prob=self.prob, top1=self.top1, gap=self.gap, ptr=ptr,
                    dist_ids=np.concatenate([d[0] for d in self.dist]).astype(np.int32),
                    dist_p=np.concatenate([d[1] for d in self.dist]).astype(np.float64),
                    sq=self.sq, maxabs=self.maxabs, first=np.int64(self.first))


def fingerprint(arrays):
    """Hash of the run-defining fields (replay determinism check)."""
    digest = hashlib.sha256()
    for key in ("visited", "decision", "own_decision", "prob", "top1", "gap", "ptr", "dist_ids", "dist_p"):
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def record_path(run_dir, seq_id):
    return Path(run_dir) / "records" / f"{seq_id}.npz"


def save_record(run_dir, seq_id, arrays):
    path = record_path(run_dir, seq_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_record(run_dir, seq_id):
    with np.load(record_path(run_dir, seq_id)) as data:
        return {k: data[k] for k in data.files}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- transient logit dumps

class LogitDump:
    """Disk-backed (page cache first) full logits of generated positions, native dtype.

    One memmap file per subset of sequences; rows are addressed by (seq id, generated
    index). Deleted after the oracle comparison consumed it.
    """

    def __init__(self, directory, sequences, vocab, dtype):
        self.dtype = dtype
        self.itemsize = torch.empty((), dtype=dtype).element_size()
        self.offsets, total = {}, 0
        for seq in sequences:
            self.offsets[seq["id"]] = total
            total += len(seq["forced"])
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"dump-{os.getpid()}.bin"
        np_dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[self.itemsize]
        self.array = np.memmap(self.path, dtype=np_dtype, mode="w+", shape=(max(total, 1), vocab))
        self.np_dtype = np_dtype
        self.written = np.zeros(max(total, 1), bool)

    def write(self, seq_id, indices, logits):
        rows = np.asarray(indices) + self.offsets[seq_id]
        raw = logits.detach().contiguous().cpu().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[self.itemsize])
        self.array[rows] = raw.numpy().view(self.np_dtype)
        self.written[rows] = True

    def read(self, seq_id, start, end, device):
        a, b = self.offsets[seq_id] + start, self.offsets[seq_id] + end
        if not self.written[a:b].all():
            raise AssertionError(f"{seq_id}: dumped logits missing for generated positions {start}..{end}")
        raw = torch.from_numpy(np.ascontiguousarray(self.array[a:b]).view(
            {2: np.int16, 4: np.int32, 8: np.int64}[self.itemsize]))
        return raw.view(self.dtype).to(device)

    def close(self):
        del self.array
        self.path.unlink(missing_ok=True)


def dump_subsets(sequences, vocab, itemsize, budget_bytes):
    """Split a block into consecutive subsets whose dumps fit the budget."""
    subsets, current, size = [], [], 0
    for seq in sequences:
        need = len(seq["forced"]) * vocab * itemsize
        if current and size + need > budget_bytes:
            subsets.append(current)
            current, size = [], 0
        current.append(seq)
        size += need
    if current:
        subsets.append(current)
    return subsets
