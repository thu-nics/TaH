"""Observational hooks and full-vocabulary reference self-noise measurements.

Used only by fixture tooling/tests, never by the engine runtime. No hook
changes a return value, and no language-model training labels are supplied.
"""
from __future__ import annotations

import torch


def pack(sequences, pad_id, device, *, side="left", extra=0, labels=None, prompt_lengths=None):
    if prompt_lengths is not None:
        if len(prompt_lengths) != len(sequences) or any(not 0 < p <= len(s) for p, s in zip(prompt_lengths, sequences)):
            raise ValueError('prompt_lengths must identify each nonempty prompt prefix')
        prompt_width = max(prompt_lengths)
        width = prompt_width + max(len(s) - p for p, s in zip(prompt_lengths, sequences))
    else:
        width = max(map(len, sequences)) + extra
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    oracle = torch.full_like(ids, -100) if labels is not None else None
    columns = []
    for row, sequence in enumerate(sequences):
        start = width - len(sequence) if side == "left" else 0
        if prompt_lengths is None:
            col = torch.arange(start, start + len(sequence), device=device)
        else:
            prefix = prompt_lengths[row]
            col = torch.cat((torch.arange(prompt_width - prefix, prompt_width, device=device),
                             torch.arange(prompt_width, prompt_width + len(sequence) - prefix, device=device)))
        columns.append(col)
        ids[row, col] = torch.as_tensor(sequence, device=device)
        mask[row, col] = 1
        if oracle is not None:
            oracle[row, col] = torch.as_tensor(labels[row], device=device)
    inputs = {"input_ids": ids, "attention_mask": mask}
    if oracle is not None:
        inputs["iter_count_labels"] = oracle
    return inputs, columns


@torch.no_grad()
def capture(model, sequences, pad_id=0, *, side="left", extra=0, labels=None,
            topk=32, hidden_positions=12, chunk_size=256, chunk_boundaries=None, prompt_lengths=None):
    """Return compact CPU traces plus transient full CPU logits for noise.

    The reference packs active tokens in row-major order before the decider
    call. Tracking that same order from its decisions maps each hook input
    back to an unpadded (sequence, position); no forward implementation is
    copied or replaced. Hooks are removed even when forward raises.
    """
    model.eval()
    inputs, columns = pack(sequences, pad_id, model.device, side=side,
                           extra=extra, labels=labels, prompt_lengths=prompt_lengths)
    decider = model._active_iter_decider
    layers = list(getattr(decider, "hidden_states_layer_nums", []))
    if hasattr(decider, "hidden_states_layer_nums") and not layers:
        layers = [model.config.num_hidden_layers]
    n = model.max_iter
    traces = []
    active = []
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        selected = torch.linspace(0, length - 1, min(length, hidden_positions)).long().unique()
        trace = {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "visited": torch.zeros(n, length, dtype=torch.bool),
            "decision": torch.zeros(n, length, dtype=torch.bool),
            "continue_probability": torch.full((n, length), float("nan"), dtype=torch.bfloat16),
            "decision_logits": torch.full((n, length), float("nan"), dtype=torch.bfloat16),
            "hidden_positions": selected,
            "hidden_layer_indices": torch.tensor(layers, dtype=torch.long),
            "decider_hidden": torch.full((n, len(selected), len(layers), model.config.hidden_size),
                                         float("nan"), dtype=model.dtype),
        }
        if labels is not None:
            trace["iter_count_labels"] = torch.tensor(labels[row], dtype=torch.long)
        traces.append(trace)
        active.extend((row, position) for position in range(length))

    def observe(module, args, kwargs, output):
        nonlocal active
        depth = kwargs["iter_depth"] - 1
        decision, raw = output
        decision = decision.detach().cpu()
        raw = raw.detach().cpu()
        if len(active) != decision.numel():
            raise AssertionError("reference active-token hook mapping changed")
        # At the cap there is no learned probability. IterLabelDecider is a
        # deterministic rule; its auxiliary neutral logits are NOT a p=0.5.
        if depth + 1 >= n or not hasattr(module, "sigmoid"):
            probability = decision.to(torch.bfloat16)
        else:
            probability = module.sigmoid(output[1]).detach().cpu()
        hidden = kwargs.get("all_hidden_states")
        if hidden is not None:
            hidden = hidden.detach().index_select(
                -2, torch.tensor(layers, device=hidden.device)).cpu()
        for index, (row, position) in enumerate(active):
            trace = traces[row]
            trace["visited"][depth, position] = True
            trace["decision"][depth, position] = decision[index]
            trace["continue_probability"][depth, position] = probability[index]
            trace["decision_logits"][depth, position] = raw[index]
            sample = (trace["hidden_positions"] == position).nonzero().flatten()
            if hidden is not None and sample.numel():
                trace["decider_hidden"][depth, sample.item()] = hidden[index]
        active = [key for index, key in enumerate(active) if decision[index]]

    width = inputs["input_ids"].shape[1]
    chunk_size = chunk_size or width
    boundaries = list(range(0, width, chunk_size)) + [width] if chunk_boundaries is None else list(chunk_boundaries)
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != width or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('chunk_boundaries must strictly increase from 0 to padded width')
    full_logits = [torch.empty(len(seq), model.config.vocab_size, dtype=model.dtype)
                   for seq in sequences]
    for trace in traces:
        trace["pass_count"] = torch.zeros_like(trace["input_ids"])
    cache = None
    hook = decider.register_forward_hook(observe, with_kwargs=True)
    try:
        for start, end in zip(boundaries, boundaries[1:]):
            active = [(row, position) for row, col in enumerate(columns)
                      for position, column in enumerate(col.tolist()) if start <= column < end]
            chunk = {"input_ids": inputs["input_ids"][:, start:end],
                     "attention_mask": inputs["attention_mask"][:, :end]}
            if labels is not None:
                chunk["iter_count_labels"] = inputs["iter_count_labels"][:, start:end]
            outputs = model(**chunk, past_key_values=cache, use_cache=True,
                            new_sequence=start == 0)
            cache = outputs.past_key_values
            for row, col in enumerate(columns):
                selected = (col >= start) & (col < end)
                local_col = col[selected] - start
                positions = selected.nonzero().flatten().cpu()
                full_logits[row][positions] = outputs.logits[row, local_col].detach().cpu()
                traces[row]["pass_count"][positions] = outputs.iter_count[row, local_col].cpu()
            del outputs
    finally:
        hook.remove()
    for trace, final in zip(traces, full_logits):
        values, indices = final.topk(min(topk, final.shape[-1]), dim=-1)
        trace["final_topk_values"] = values
        trace["final_topk_indices"] = indices.to(torch.int32)
        # topk ties need not put argmax first, especially in bf16.
        trace["final_top1"] = final.argmax(-1).to(torch.int32)
        # A few dense vocabulary rows let engine tests measure full-vocab
        # error directly from disk, while every position retains its top-k.
        trace["logit_positions"] = torch.linspace(0, len(final) - 1, min(4, len(final))).long().unique()
        trace["final_logits_sample"] = final[trace["logit_positions"]].clone()
        if not torch.equal(trace["pass_count"], trace["visited"].sum(0)):
            raise AssertionError("hook depth coverage disagrees with reference iter_count")
    return traces, full_logits


def compare(reference, candidate, reference_logits, candidate_logits, threshold):
    """Full vocab error in fp32; decisions include non-common visited depths.

    Missing depths count as disagreements (rather than hiding cascades).
    Threshold margins are reported only where both sides made MLP decisions.
    """
    both = reference["visited"] & candidate["visited"]
    union = reference["visited"] | candidate["visited"]
    mismatch = union & (~both | (reference["decision"] != candidate["decision"]))
    common_mismatch = both & (reference["decision"] != candidate["decision"])
    margin = (reference["continue_probability"].float() - threshold).abs()
    per_depth = []
    for depth in range(len(union)):
        count = int(union[depth].sum())
        errors_at_depth = int(mismatch[depth].sum())
        per_depth.append({"depth": depth + 1, "decision_compared": count,
                          "decision_mismatches": errors_at_depth,
                          "decision_agreement": 1 - errors_at_depth / max(count, 1),
                          "visited_mismatches": int((union[depth] & ~both[depth]).sum()),
                          "common_decision_mismatches": int(common_mismatch[depth].sum()),
                          "mismatches_within_0.01": int((common_mismatch[depth] & (margin[depth] <= 0.01)).sum())})
    errors = logit_error(reference_logits, candidate_logits)
    fixed_topk = logit_error(reference["final_topk_values"],
                            candidate_logits.gather(-1, reference["final_topk_indices"].long()))
    sampled = logit_error(reference["final_logits_sample"],
                         candidate_logits[reference["logit_positions"]])
    return {
        "decision_compared": int(union.sum()),
        "decision_mismatches": int(mismatch.sum()),
        "decision_agreement": 1 - float(mismatch.sum()) / max(int(union.sum()), 1),
        "visited_mismatches": int((union & ~both).sum()),
        "common_decision_mismatches": int(common_mismatch.sum()),
        "mismatches_within_0.01": int((common_mismatch & (margin <= 0.01)).sum()),
        **errors,
        "retained_topk_error": fixed_topk,
        "retained_dense_error": sampled,
        "top1_compared": len(reference_logits),
        "top1_mismatches": int((reference_logits.argmax(-1) != candidate_logits.argmax(-1)).sum()),
        "top1_agreement": float((reference_logits.argmax(-1) == candidate_logits.argmax(-1)).float().mean()),
        "pass_count_mismatches": int((reference["pass_count"] != candidate["pass_count"]).sum()),
        "per_depth": per_depth,
    }


def logit_error(reference, candidate):
    """FP32 errors accumulated in bounded blocks; no full fp32 vocab copy."""
    elements, absolute, squared, maximum = 0, 0.0, 0.0, 0.0
    for start in range(0, len(reference), 128):
        diff = reference[start:start + 128].float() - candidate[start:start + 128].float()
        elements += diff.numel()
        absolute += float(diff.abs().double().sum())
        squared += float(diff.square().double().sum())
        maximum = max(maximum, float(diff.abs().max()))
    return {"logit_elements": elements, "logit_abs_sum": absolute,
            "logit_squared_sum": squared, "logit_max_abs": maximum,
            "logit_mae": absolute / max(elements, 1),
            "logit_rmse": (squared / max(elements, 1)) ** 0.5}


def divergence(reference, candidate):
    """Zero-based first divergent output token, including unequal EOS length."""
    for position, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return position
    return min(len(reference), len(candidate)) if len(reference) != len(candidate) else None


def summarize(rows):
    totals = {key: sum(row[key] for row in rows) for key in (
        "decision_compared", "decision_mismatches", "visited_mismatches",
        "common_decision_mismatches", "mismatches_within_0.01", "logit_elements",
        "logit_abs_sum", "logit_squared_sum", "top1_compared", "top1_mismatches",
        "pass_count_mismatches")}
    totals.update(
        decision_agreement=1 - totals["decision_mismatches"] / max(totals["decision_compared"], 1),
        logit_mae=totals["logit_abs_sum"] / max(totals["logit_elements"], 1),
        logit_rmse=(totals["logit_squared_sum"] / max(totals["logit_elements"], 1)) ** 0.5,
        logit_max_abs=max((row["logit_max_abs"] for row in rows), default=0),
        top1_agreement=1 - totals["top1_mismatches"] / max(totals["top1_compared"], 1),
    )
    # Gate targets use error / disagreement rate, not 1.5 * agreement.
    totals["g_tf_logit_mae_limit"] = 1.5 * totals["logit_mae"]
    totals["g_tf_logit_rmse_limit"] = 1.5 * totals["logit_rmse"]
    totals["g_tf_logit_max_abs_limit"] = 1.5 * totals["logit_max_abs"]
    totals["g_tf_top1_disagreement_limit"] = 1.5 * (1 - totals["top1_agreement"])
    totals["per_depth"] = []
    for depth in range(max((len(row["per_depth"]) for row in rows), default=0)):
        relevant = [row["per_depth"][depth] for row in rows]
        combined = {key: sum(row[key] for row in relevant) for key in (
            "decision_compared", "decision_mismatches", "visited_mismatches",
            "common_decision_mismatches", "mismatches_within_0.01")}
        combined["depth"] = depth + 1
        combined["decision_agreement"] = 1 - combined["decision_mismatches"] / max(combined["decision_compared"], 1)
        totals["per_depth"].append(combined)
    for metric in ("retained_topk_error", "retained_dense_error"):
        retained = [row[metric] for row in rows]
        count = sum(row["logit_elements"] for row in retained)
        mae = sum(row["logit_abs_sum"] for row in retained) / max(count, 1)
        rmse = (sum(row["logit_squared_sum"] for row in retained) / max(count, 1)) ** 0.5
        maximum = max((row["logit_max_abs"] for row in retained), default=0)
        totals[metric] = {"logit_elements": count, "logit_mae": mae,
                          "logit_rmse": rmse, "logit_max_abs": maximum,
                          "g_tf_logit_mae_limit": 1.5 * mae,
                          "g_tf_logit_rmse_limit": 1.5 * rmse,
                          "g_tf_logit_max_abs_limit": 1.5 * maximum}
    return totals
