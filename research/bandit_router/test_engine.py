"""Counterfactual isolation, native accumulated logits, and engine policy wiring."""
import pytest
import torch
from torch.nn import functional as F

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.model import EagerTaHModel
from tah.engine.tests.test_model import reference
from .core import BanditRouter, features
from .engine import EnginePairs, EngineRoutingDecider


@pytest.mark.parametrize('threshold', [-1., 2., .5])
def test_pairs_match_independent_forced_branches_and_preserve_history(tmp_path, threshold):
    torch.set_num_threads(2)
    reference(tmp_path, max_iter=2, label=False)
    model = EagerTaHModel.from_pretrained(tmp_path).eval()
    (model.eval_iter_decider or model.iter_decider).threshold.data.fill_(threshold)
    pairs = EnginePairs(model, 64)
    history = pairs.start([3, 7, 11], 16)
    prefix = [3, 7]
    for position, (token, target) in enumerate([(11, 13), (13, 17), (17, 19)], 2):
        state, losses, depth = pairs.step(token, target, position)
        for action in (1, 2):
            independent = EnginePairs(model, 64)
            request = independent.pool.create_request('branch', context_limit=16)
            independent.request = request
            try:
                for i, (old, old_depth) in enumerate(zip(prefix, history)):
                    model.forward_chunk([old], independent.pool, request,
                        start_position=i, iter_count_labels=[old_depth], decode=True)
                original = model.iter_decider, model.eval_iter_decider
                native = model.eval_iter_decider or model.iter_decider
                model.iter_decider = model.eval_iter_decider = EngineRoutingDecider(native, pairs.layers,
                    policy='stop' if action == 1 else 'continue')
                output = model.forward_chunk([token], independent.pool, request,
                    start_position=position, hidden_positions=[0], decode=True)
                assert int(output.iter_count.item()) == action
                expected = F.cross_entropy(output.logits.float(), torch.tensor([target]))
                torch.testing.assert_close(losses[action - 1], expected, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(state[:-2].reshape(3, -1), output.decider_hidden[0, 0].float(), rtol=0, atol=0)
                model.iter_decider, model.eval_iter_decider = original
            finally:
                independent.release()
        # Speculative depth-2 KV must exist only for imitation continuations.
        assert (position in pairs.request.streams[1].host_positions) == (depth == 2)
        assert torch.isfinite(state).all() and torch.isfinite(losses).all()
        prefix.append(token)
        history.append(depth)
    pairs.release()


@pytest.mark.parametrize('policy,expected', [('stop', 1), ('continue', 2)])
def test_engine_installs_policy_before_prefill_and_decode(tmp_path, policy, expected):
    reference(tmp_path, max_iter=2, label=False)
    def factory(model):
        return EngineRoutingDecider(model.iter_decider, (1, 1, 2), policy=policy)
    with TaHEngine(tmp_path, device='cpu', kv_capacity=64, max_batch_size=1,
                   host_swap_bytes=0, overlap=False, decider_factory=factory) as engine:
        result = engine.generate([Request([3, 7, 11], 3)], SamplingParams(temperature=0))[0]
        assert result.prompt_pass_counts == [expected] * 3
        assert result.generated_pass_counts == [expected] * 3
        assert engine.pool.audit()['owned'] == 0


def test_entropy_and_bandit_caps_and_feature_order():
    torch.manual_seed(42)
    logits = torch.randn(3, 17)
    hidden = torch.randn(3, 5, 8)
    layers = (1, 2, 4)
    state = features(logits, hidden, layers)
    torch.testing.assert_close(state, features(logits, hidden, layers, indices=torch.tensor(layers)), rtol=0, atol=0)
    router = BanditRouter(state.shape[1])
    for policy in ('entropy', 'bandit'):
        decider = EngineRoutingDecider(None, layers, policy, router, threshold=-1., logit_threshold=-float('inf'))
        keep, _, _ = decider(logits, torch.tensor([1, 2, 1]), hidden)
        assert keep.tolist() == [True, False, True]


def test_collection_resume_and_unbranched_replay(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from . import engine_run

    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    reference(model_dir, max_iter=2, label=False)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [3, 7, 11]

        def encode(self, *args, **kwargs):
            return [13, 17, 19]

    def load(args):
        model = EagerTaHModel.from_pretrained(args.model).eval()
        (model.eval_iter_decider or model.iter_decider).threshold.data.fill_(args.imitation_threshold)
        return model, Tokenizer()

    monkeypatch.setattr(engine_run, 'load_engine_model', load)
    refs = tmp_path / 'references.jsonl'
    refs.write_text('\n'.join(json.dumps(dict(id=str(i), question=f'problem {i}', reference='answer')) for i in range(2)))
    args = SimpleNamespace(model=str(model_dir), device='cpu', references=refs,
        output_dir=tmp_path / 'run', seed=42, imitation_threshold=.5, max_reference_tokens=8)
    engine_run.collect(args)
    engine_run.collect(args)
    cache = args.output_dir / 'cache.pt'
    engine_run.verify(SimpleNamespace(cache=cache, device='cpu', max_problems=0))
    report = json.loads(cache.with_suffix('.verification.json').read_text())
    assert report['verified_rows'] == 6
    assert report['verified_problems'] == 2
    assert report['depth1_hidden_features'] == 'bit_exact_at_storage_dtype'
