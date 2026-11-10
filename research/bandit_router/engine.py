"""Bandit policies and paired teacher forcing on the engine's persistent KV pool."""
import torch
from torch import nn
from torch.nn import functional as F

from tah.engine.attention import AttentionRow, DecodeBucket
from tah.engine.kv_cache import SlotPool
from .core import features


class EngineRoutingDecider(nn.Module):
    """Graph-safe engine contract; the policy also applies to prompt tokens."""

    def __init__(self, imitation, layers, policy='imitation', router=None,
                 threshold=0.5, logit_threshold=None):
        super().__init__()
        if policy not in ('imitation', 'stop', 'continue', 'entropy', 'bandit'):
            raise ValueError(f'Unknown routing policy: {policy}')
        if policy == 'bandit' and router is None:
            raise ValueError('bandit policy requires a router')
        self.imitation, self.layers, self.policy = imitation, tuple(layers), policy
        self.router = router
        self.threshold = float(threshold)
        self.logit_threshold = logit_threshold
        indices = None
        if hasattr(imitation, 'selected_layers'):
            indices = torch.tensor(imitation.selected_layers(max(layers)),
                                   device=imitation.threshold.device, dtype=torch.long)
        self.register_buffer('native_indices', indices)
        device = router.mean.device if router is not None else (
            imitation.threshold.device if imitation is not None else 'cpu')
        self.register_buffer('feature_indices', torch.tensor(layers, device=device, dtype=torch.long))

    def selected_layers(self, layer_count):
        return list(self.layers)

    def forward(self, logits, depth, hidden=None, labels=None, *, hidden_indices=None):
        # The native decider keeps its own indices, distinct from router features.
        if self.policy == 'imitation':
            return self.imitation(logits, depth, hidden, labels, hidden_indices=self.native_indices)
        cap = torch.as_tensor(depth, device=logits.device) >= 2
        if self.policy in ('stop', 'continue'):
            chosen = torch.full_like(logits[..., 0], self.policy == 'continue', dtype=torch.bool)
            raw = torch.where(chosen, 10., -10.).to(logits.dtype)
        else:
            state = features(logits, hidden, self.layers, indices=self.feature_indices)
            if self.policy == 'entropy':
                raw = state[..., -2]
                chosen = raw > self.threshold
            else:
                raw = self.router(state)
                chosen = (raw > self.logit_threshold if self.logit_threshold is not None
                          else raw.sigmoid() > self.threshold)
        chosen = chosen & ~cap
        raw = torch.where(cap, -10., raw)
        probability = torch.where(cap, 0., raw.sigmoid()).to(logits.dtype)
        return chosen, raw.to(logits.dtype), probability


class EnginePairs:
    """Compute both actions, retaining only frozen imitation history.

    Depth 2 writes one speculative stream entry. If imitation stops, pop that
    entry before the next token; no past KV tensors are copied or recomputed.
    """

    def __init__(self, model, capacity):
        if model.max_iter != 2:
            raise ValueError('Bandit training requires a depth-2 checkpoint')
        self.model = model
        cfg = model.config
        n = cfg['num_hidden_layers']
        self.layers = (1, max(1, n // 2), n)
        self.pool = SlotPool(capacity, n, cfg['num_key_value_heads'], cfg['head_dim'],
                             2, device=model.device, dtype=model.dtype)
        self.attention = DecodeBucket(self.pool, 1, capacity, cfg['num_attention_heads'])
        self.request = None

    def start(self, prompt, context_limit):
        if self.request is not None:
            raise RuntimeError('Release the previous problem first')
        self.request = self.pool.create_request('problem', context_limit=context_limit)
        history = []
        # Same prefill layout for collection and unbranched replay.
        for start in range(0, len(prompt) - 1, 128):
            output = self.model.forward_chunk(prompt[start:min(start + 128, len(prompt) - 1)],
                self.pool, self.request, start_position=start)
            history.extend(output.iter_count.tolist())
        return history

    @torch.inference_mode()
    def step(self, token, target, position, paired=True):
        model = self.model
        ids = torch.tensor([token], device=model.device)
        pos = torch.tensor([position], device=model.device)

        def run(embedding, depth):
            slots = self.pool.allocate(self.request, depth - 1, [position])
            self.attention.prepare([AttentionRow(self.request, position, depth)])
            return model.forward_cached(embedding, pos, torch.full_like(pos, depth),
                                        slots, self.pool, self.attention)

        first = run(model.model.embed_tokens(ids, False), 1)
        decider = model.eval_iter_decider or model.iter_decider
        keep, _, _ = decider(first.logits, 1, first.hidden_states)
        depth = 2 if bool(keep.item()) else 1
        state = features(first.logits, first.hidden_states, self.layers)[0].cpu()
        target = torch.tensor([target], device=model.device)
        loss1 = F.cross_entropy(first.logits.float(), target)
        if paired or depth == 2:
            values, indices = first.logits.topk(min(model.input_topk, first.logits.shape[-1]), dim=-1)
            embedding = (values.softmax(-1)[..., None] * model.model.embed_tokens.weight[indices]).sum(-2).to(model.dtype)
            second = run(embedding, 2)
            # Preserve the native bf16 additive-logit arithmetic.
            loss2 = F.cross_entropy((first.logits + second.logits).float(), target)
            if depth == 1:
                self.pool.pop_last(self.request, 1, position)
        else:
            loss2 = torch.full_like(loss1, float('nan'))
        return state, torch.stack((loss1, loss2)).cpu(), depth

    def release(self):
        if self.request is not None:
            self.pool.release(self.request)
            self.request = None
        assert self.pool.audit()['owned'] == 0
