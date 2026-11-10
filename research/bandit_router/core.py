"""Depth-1 states, isolated counterfactuals, and the proposal's exact objective."""
from contextlib import contextmanager
import copy
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def features(logits, hidden, layers):
    """HF hidden-state indices: 0 is embeddings, 1 is first transformer layer."""
    if hidden is None or max(layers) >= hidden.shape[-2] or min(layers) < 1:
        raise ValueError("Expected shallow/middle/final transformer hidden states")
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1, keepdim=True)
    maximum = p.max(-1, keepdim=True).values
    return torch.cat((hidden[..., list(layers), :].flatten(-2).float(), entropy, maximum), -1)


class RoutingDecider(nn.Module):
    """Use the wrapper's native forward loop; never reconstruct depth-2 logits."""

    def __init__(self, imitation, layers, policy="imitation", router=None, threshold=0.5, logit_threshold=None):
        super().__init__()
        self.imitation = imitation
        self.layers = tuple(layers)
        # This attribute tells TaH to request all backbone hidden states.
        self.hidden_states_layer_nums = list(layers)
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.policy = policy
        self.router = router
        self.logit_threshold = logit_threshold
        self.state = None
        self.imitation_decision = None
        self.branch_cache = None
        self.capture_branch = False
        self.stop_logits = None
        self.stop_cache = None

    def forward(self, logits, iter_depth, all_hidden_states=None, **kwargs):
        if iter_depth >= 2:
            return torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device), torch.full_like(logits[..., 0], -10)
        state = features(logits, all_hidden_states, self.layers)
        decision, original_logits = self.imitation(
            logits=logits, iter_depth=iter_depth, all_hidden_states=all_hidden_states, **kwargs
        )
        self.state = state.detach().clone()
        self.imitation_decision = decision.detach().clone()
        if self.capture_branch:
            self.stop_logits = logits.detach().clone()
            # Cache already contains the current token's depth-1 KV. Snapshot
            # it before the wrapper starts depth 2, only if imitation stops.
            self.stop_cache = fork_cache(self.branch_cache) if not bool(decision.item()) else None
        if self.policy == "imitation":
            return decision, original_logits
        if self.policy in ("stop", "continue"):
            chosen = torch.full_like(decision, self.policy == "continue")
            return chosen, torch.where(chosen, 10.0, -10.0)
        if self.policy == "entropy":
            score = state[..., -2]
        elif self.policy == "bandit":
            policy_logits = self.router(state)
            if self.logit_threshold is not None:
                return policy_logits > self.logit_threshold, policy_logits
            score = policy_logits.sigmoid()
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        chosen = score > self.threshold
        return chosen, torch.logit(score.clamp(1e-6, 1 - 1e-6)) if self.policy == "bandit" else score


@contextmanager
def override_decider(model, decider):
    original = model.iter_decider, model.eval_iter_decider
    try:
        model.iter_decider = decider
        model.eval_iter_decider = decider
        yield
    finally:
        model.iter_decider, model.eval_iter_decider = original


def fork_cache(cache):
    """Clone every KV tensor and metadata field, including HF cache internals."""
    return copy.deepcopy(cache)


@torch.inference_mode()
def paired_step(model, decider, cache, input_id, target_id):
    """Both branches start before the current token, with identical past routing.

    Only the branch chosen by the frozen imitation decider is committed. The
    losing counterfactual cache is discarded before the next reference token.
    No labels enter model.forward (which otherwise can force continuation).
    """
    states, outputs, decisions = [], [], []
    try:
        for action in ("stop", "continue"):
            decider.policy = action
            output = model(input_ids=input_id, past_key_values=fork_cache(cache), use_cache=True)
            states.append(decider.state.clone())
            decisions.append(decider.imitation_decision.clone())
            outputs.append(output)
        if not torch.equal(states[0], states[1]) or not torch.equal(decisions[0], decisions[1]):
            raise RuntimeError("Counterfactual branches have different depth-1 histories")
        if states[0].shape[0] != 1 or decisions[0].numel() != 1:
            raise ValueError("Tiny cache collector expects one current token in one problem")
        losses = [F.cross_entropy(o.logits[:, -1].float(), target_id.reshape(1)) for o in outputs]
        depth = 2 if bool(decisions[0].item()) else 1
        if [int(o.iter_count.item()) for o in outputs] != [1, 2]:
            raise RuntimeError("STOP/CONTINUE did not execute depths 1/2")
        return states[0][0].cpu(), torch.stack(losses).cpu(), depth, outputs[depth - 1].past_key_values
    finally:
        decider.policy = "imitation"


@torch.inference_mode()
def paired_step_fast(model, decider, cache, input_id, target_id):
    """Share depth 1, then retain its STOP snapshot or native CONTINUE cache.

    The decider callback runs after depth-1 KV writes and before depth 2.
    STOP logits and cache are captured there; CONTINUE uses the wrapper's
    actual accumulated logits. Never commit the forced branch to history.
    """
    from tah.model.causal_cache import TaHCache
    run_cache = fork_cache(cache) if cache is not None else TaHCache().to(model.device)
    decider.branch_cache = run_cache
    decider.capture_branch = True
    decider.policy = "continue"
    try:
        output = model(input_ids=input_id, past_key_values=run_cache, use_cache=True)
        if int(output.iter_count.item()) != 2:
            raise RuntimeError("Forced CONTINUE did not execute depth 2")
        pair = torch.stack((
            F.cross_entropy(decider.stop_logits.float(), target_id.reshape(1)),
            F.cross_entropy(output.logits[:, -1].float(), target_id.reshape(1)),
        ))
        depth = 2 if bool(decider.imitation_decision.item()) else 1
        committed = output.past_key_values if depth == 2 else decider.stop_cache
        return decider.state[0].cpu(), pair.cpu(), depth, committed
    finally:
        decider.capture_branch = False
        decider.branch_cache = decider.stop_cache = decider.stop_logits = None
        decider.policy = "imitation"


class BanditRouter(nn.Module):
    def __init__(self, feature_dim, width=64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("scale", torch.ones(feature_dim))
        self.net = nn.Sequential(nn.Linear(feature_dim, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, state):
        return self.net((state.float() - self.mean) / self.scale).squeeze(-1)


def bandit_loss(logits, action_losses, penalty):
    if penalty < 0:
        raise ValueError("Compute penalty must be nonnegative")
    reward = action_losses[:, 0] - action_losses[:, 1] - penalty
    return -(logits.sigmoid() * reward).mean()


def problem_split(records, seed=42, validation_fraction=0.1):
    """Split entire problems; repeated problem text stays in one group."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    keys = [digest(" ".join(r["question"].lower().split())) for r in records]
    groups = sorted(set(keys), key=lambda k: digest([seed, k]))
    if len(groups) < 2:
        raise ValueError("Need at least two distinct problems for train/validation")
    n = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation = set(groups[:n])
    return torch.tensor([k not in validation for k in keys]), torch.tensor([k in validation for k in keys])


def calibrate_budget(scores, losses, fraction, penalty=0.0):
    """Calibrate strict score > threshold in O(N log N), respecting score ties.

    Use raw policy logits, so fp32 sigmoid saturation cannot erase rankings.
    Among equally close budgets, use held-out reward to break ties.
    """
    if not 0 <= fraction <= 1 or scores.ndim != 1 or not len(scores):
        raise ValueError("Expected nonempty scores and a budget fraction in [0,1]")
    sorted_scores, order = scores.sort(descending=True)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    eligible = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    distance = (eligible.float() / len(scores) - fraction).abs()
    candidates = eligible[distance <= distance.min() + 1e-7]
    rewards = losses[:, 0] - losses[:, 1] - penalty
    cumulative = torch.cat((torch.zeros(1), rewards[order].cumsum(0)))
    k = int(candidates[cumulative[candidates].argmax()])
    threshold = float(sorted_scores[k]) if k < len(scores) else float("-inf")
    chosen = scores > threshold
    assert int(chosen.sum()) == k
    return threshold, chosen
