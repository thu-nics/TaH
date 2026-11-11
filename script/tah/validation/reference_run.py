"""Teacher-forced runs of the unmodified reference model in any precision.

Layouts:
  * ``generate`` - the evaluation layout of ``TaHForCasualLM_generate``: a batch of
    sequences, prompts left-padded and prefilled in one forward, then one cached
    single-token forward per step with the generated columns aligned. A sequence
    that has ended is fed padding (mask 0) instead of its post-EOS samples; that only
    changes its own discarded row.
  * ``chunk`` - one sequence, cached forwards over fixed-width chunks (oracle layout).

Decisions are observed with a forward hook on the decider (as
``tah.engine.reference_capture.capture`` does) and, in forced mode, replaced by the
target's decisions in the hook's return value; the model code is not modified.
"""
from __future__ import annotations

import numpy as np
import torch

from tah.engine.reference_capture import pack

from .common import Recorder, effective_threshold, inputs_of, reference_distribution, top_gap
from .fp64_attention import fp64_attention_guard


class ReferenceRunner:
    def __init__(self, model):
        self.model = model.eval()
        self.decider = model._active_iter_decider
        self.max_iter = model.max_iter
        self.threshold = effective_threshold(self.decider)

    def run(self, seqs, **kwargs):
        """Return one Recorder per sequence (keyword arguments: ``_run``).

        A float64 model runs inside ``fp64_attention_guard`` (PyTorch #143644); every
        such run must route attention through it.
        """
        if self.model.dtype != torch.float64:
            return self._run(seqs, **kwargs)
        with fp64_attention_guard() as guard:
            before = guard.calls
            recorders = self._run(seqs, **kwargs)
        if guard.calls == before:
            raise AssertionError("float64 reference attention bypassed F.scaled_dot_product_attention; "
                                 "the PyTorch #143644 workaround did not apply")
        return recorders

    @torch.no_grad()
    def _run(self, seqs, *, layout, chunk=256, pad_id=0, forced=None, consumer=None,
             distribution=reference_distribution):
        """forced: per sequence a bool array [max_iter, L] of decisions to impose, or None.

        consumer(seq, generated_indices, logits): sees every generated position's final
        logits (model dtype) once, e.g. to dump or compare them.
        """
        model = self.model
        inputs_list = [inputs_of(s) for s in seqs]
        sequences = [ids for ids, _ in inputs_list]
        recorders = [Recorder(s, self.max_iter) for s in seqs]
        if layout == "generate":
            prompts = [len(s["prompt"]) for s in seqs]
            inputs, columns = pack(sequences, pad_id, model.device, side="left", prompt_lengths=prompts)
            width, prompt_width = inputs["input_ids"].shape[1], max(prompts)
            boundaries = [0] + list(range(prompt_width, width + 1))
        elif layout == "chunk":
            if len(seqs) != 1:
                raise ValueError("chunk layout runs one sequence at a time")
            inputs, columns = pack(sequences, pad_id, model.device, side="left")
            width = inputs["input_ids"].shape[1]
            boundaries = list(range(0, width, chunk)) + [width]
        else:
            raise ValueError(f"unknown reference layout {layout!r}")
        if forced is not None and len(forced) != len(seqs):
            raise ValueError("forced decisions need one entry per sequence")
        column_arrays = [c.cpu().numpy() for c in columns]

        def in_chunk(row, start, end):
            col = column_arrays[row]
            return range(int(np.searchsorted(col, start)), int(np.searchsorted(col, end)))
        active = []

        def observe(module, args, kwargs, output):
            nonlocal active
            depth = kwargs["iter_depth"]
            decision, raw = output
            own = decision.detach().cpu().numpy().astype(bool)
            if len(active) != own.size:
                raise AssertionError("reference active-token mapping changed")
            if depth < self.max_iter and hasattr(module, "sigmoid"):
                prob = module.sigmoid(raw).detach().double().cpu().numpy()
            else:
                prob = np.full(own.size, np.nan)
            if forced is not None:
                chosen = np.array([forced[row][depth - 1, pos] for row, pos in active], bool)
                if depth >= self.max_iter and chosen.any():
                    raise AssertionError("forced continuation past the pass cap")
            else:
                chosen = own
            for k, (row, pos) in enumerate(active):
                recorders[row].decide(depth, pos, bool(chosen[k]), bool(own[k]), float(prob[k]))
            active = [key for key, keep in zip(active, chosen) if keep]
            if forced is not None:
                return torch.as_tensor(chosen, device=decision.device), raw
            return None

        hook = self.decider.register_forward_hook(observe, with_kwargs=True)
        cache = None
        try:
            for start, end in zip(boundaries, boundaries[1:]):
                spans = [in_chunk(row, start, end) for row in range(len(seqs))]
                active = [(row, pos) for row, span in enumerate(spans) for pos in span]
                if not active:
                    continue
                batch = {"input_ids": inputs["input_ids"][:, start:end],
                         "attention_mask": inputs["attention_mask"][:, :end]}
                outputs = model(**batch, past_key_values=cache, use_cache=True, new_sequence=start == 0)
                cache = outputs.past_key_values
                rows, cols, owners = [], [], []
                counts_all = outputs.iter_count.cpu().numpy()
                for row, span in enumerate(spans):
                    if not len(span):
                        continue
                    first = inputs_list[row][1]
                    local = column_arrays[row][span.start:span.stop] - start
                    seen = recorders[row].visited[:, span.start:span.stop].sum(0)
                    if not np.array_equal(counts_all[row, local], seen):
                        raise AssertionError("hook depth coverage disagrees with reference iter_count")
                    for pos, c in zip(span, local.tolist()):
                        if pos >= first:
                            rows.append(row)
                            cols.append(c)
                            owners.append((row, pos))
                if rows:
                    final = outputs.logits[torch.tensor(rows, device=model.device),
                                           torch.tensor(cols, device=model.device)]
                    self._finals(seqs, recorders, owners, final, consumer, distribution)
                del outputs
        finally:
            hook.remove()
            del cache
        return recorders

    @staticmethod
    def _finals(seqs, recorders, owners, final, consumer, distribution):
        top1, gap = top_gap(final)
        dist = distribution(final)
        by_row = {}
        for k, (row, pos) in enumerate(owners):
            by_row.setdefault(row, []).append((k, pos))
        for row, items in by_row.items():
            index = [k for k, _ in items]
            positions = [pos for _, pos in items]
            recorders[row].final(positions, top1[index], gap[index], [dist[k] for k in index])
            if consumer is not None:
                first = recorders[row].first
                consumer(seqs[row], [p - first for p in positions],
                         final[torch.tensor(index, device=final.device)])
