"""Collect, replay and generate with the TaH v1 inference engine."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .collect_large import atomic_save
from .core import BanditRouter, digest
from .engine import EnginePairs, EngineRoutingDecider
from .run import load_references


def source_hashes():
    paths = sorted(Path('tah/engine').glob('*.py')) + [Path(__file__), Path(__file__).with_name('engine.py'), Path(__file__).with_name('core.py')]
    return {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def model_hashes(path):
    paths = sorted(Path(path).rglob('*.safetensors')) + sorted(Path(path).rglob('*.json'))
    if not paths:
        raise ValueError('Expected a local checkpoint directory')
    result = {}
    for p in paths:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        result[str(p.relative_to(path))] = h.hexdigest()
    return result


def load_engine_model(args):
    from tah.engine.model import EagerTaHModel
    from transformers import AutoTokenizer
    model = EagerTaHModel.from_pretrained(args.model, device=args.device).eval().requires_grad_(False)
    if model.max_iter != 2:
        raise ValueError('Expected a depth-2 TaH checkpoint')
    (model.eval_iter_decider or model.iter_decider).threshold.data.fill_(args.imitation_threshold)
    return model, AutoTokenizer.from_pretrained(args.model)


def collect(args):
    records = load_references(args.references)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(backend='tah_engine', model=args.model, imitation_threshold=args.imitation_threshold,
        reference_sha256=digest(records), seed=args.seed, max_reference_tokens=args.max_reference_tokens,
        source_sha256=source_hashes(), checkpoint_sha256=model_hashes(args.model),
        history_policy='frozen_imitation', loss2='native_additive_logits', feature_storage='float16',
        truncated=False, collection_cuda_graphs=False, prefill_chunk_size=128)
    manifest = root / 'collection.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Collection settings changed; use a new directory')
    manifest.write_text(json.dumps(settings, indent=2))
    model, tokenizer = load_engine_model(args)
    # Prompt <=512 plus complete reference <=cap; validate actual lengths below.
    pairs = EnginePairs(model, 2 * (512 + args.max_reference_tokens))
    metadata = dict(settings, hidden_layers=list(pairs.layers), hidden_size=model.config['hidden_size'],
                    feature_order='shallow,middle,final,entropy,max_probability', torch_version=str(torch.__version__))
    shards = root / 'shards'
    shards.mkdir(exist_ok=True)
    started = time.perf_counter()
    total = 0
    with torch.inference_mode():
        for number, record in enumerate(records, 1):
            path = shards / (digest(record['id'])[:20] + '.pt')
            if path.exists():
                shard = torch.load(path, map_location='cpu', weights_only=True)
                if shard['problem']['id'] != record['id'] or shard['metadata'] != metadata:
                    raise ValueError('Mismatched engine shard')
                total += len(shard['rows'])
                continue
            prompt = tokenizer.apply_chat_template([dict(role='user', content=record['question'])],
                                                    tokenize=True, add_generation_prompt=True)
            reference = tokenizer.encode(record['reference'], add_special_tokens=False)
            if not prompt or len(prompt) > 512 or not 1 <= len(reference) <= args.max_reference_tokens:
                raise ValueError('Invalid prompt/reference length; refusing truncation')
            limit = len(prompt) + len(reference)
            if limit > model.config['max_position_embeddings']:
                raise ValueError('Problem exceeds checkpoint context')
            states, losses, rows = [], [], []
            try:
                history = pairs.start(prompt, limit)
                prefix, current = prompt[:-1].copy(), prompt[-1]
                for index, target in enumerate(reference):
                    state, loss, depth = pairs.step(current, target, len(prefix))
                    states.append(state.half())
                    losses.append(loss)
                    rows.append(dict(problem_id=record['id'], reference_index=index, position=len(prefix),
                        input_id=current, target_id=target, history_sha256=digest(dict(tokens=prefix, depths=history)),
                        imitation_depth=depth))
                    prefix.append(current)
                    history.append(depth)
                    current = target
                problem = dict(record, prompt_token_ids=prompt, reference_token_ids=reference,
                               imitation_history_depths=history)
                atomic_save(dict(features=torch.stack(states), action_losses=torch.stack(losses),
                                 rows=rows, problem=problem, metadata=metadata), path)
            finally:
                pairs.release()
            total += len(rows)
            progress = dict(stage='collecting_engine', completed_problems=number, total_problems=len(records),
                            rows=total, seconds_this_run=time.perf_counter() - started)
            (root / 'progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    states, losses, rows, problems = [], [], [], []
    for record in records:
        shard = torch.load(shards / (digest(record['id'])[:20] + '.pt'), map_location='cpu', weights_only=True)
        first = len(rows)
        rows.extend(shard['rows'])
        states.append(shard['features'])
        losses.append(shard['action_losses'])
        problems.append(dict(shard['problem'], row_start=first, row_end=len(rows)))
    atomic_save(dict(schema_version=1, features=torch.cat(states), action_losses=torch.cat(losses),
                     rows=rows, problems=problems, metadata=metadata), root / 'cache.pt')
    print(json.dumps(dict(stage='cache_complete', problems=len(problems), rows=len(rows))), flush=True)


def verify(args):
    saved = torch.load(args.cache, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if meta['backend'] != 'tah_engine' or meta['source_sha256'] != source_hashes():
        raise ValueError('Engine collection provenance differs from replay')
    if meta['checkpoint_sha256'] != model_hashes(meta['model']):
        raise ValueError('Checkpoint changed since collection')
    model, _ = load_engine_model(SimpleNamespace(model=meta['model'], device=args.device,
                                                 imitation_threshold=meta['imitation_threshold']))
    native = model.eval_iter_decider or model.iter_decider
    model.iter_decider = model.eval_iter_decider = EngineRoutingDecider(
        native, meta['hidden_layers'], policy='imitation')
    pool = EnginePairs(model, 2 * (512 + meta['max_reference_tokens']))
    checked = 0
    problems = saved['problems'][:args.max_problems or None]
    with torch.inference_mode():
        for problem in problems:
            prompt, reference = problem['prompt_token_ids'], problem['reference_token_ids']
            try:
                history = pool.start(prompt, len(prompt) + len(reference))
                prefix, current = prompt[:-1].copy(), prompt[-1]
                for index, target in enumerate(reference):
                    row_index = problem['row_start'] + index
                    row = saved['rows'][row_index]
                    assert row['history_sha256'] == digest(dict(tokens=prefix, depths=history))
                    # Independent unbranched native forward_chunk path.
                    output = model.forward_chunk([current], pool.pool, pool.request,
                        start_position=len(prefix), hidden_positions=[0], decode=True)
                    depth = int(output.iter_count.item())
                    assert depth == row['imitation_depth']
                    expected_hidden = saved['features'][row_index, :-2].reshape(len(meta['hidden_layers']), -1)
                    torch.testing.assert_close(output.decider_hidden[0, 0].cpu().to(expected_hidden.dtype),
                                               expected_hidden, rtol=0, atol=0)
                    loss = F.cross_entropy(output.logits.float(), torch.tensor([target], device=model.device))
                    torch.testing.assert_close(loss.cpu(), saved['action_losses'][row_index, depth - 1], rtol=1e-5, atol=1e-5)
                    prefix.append(current)
                    history.append(depth)
                    current = target
                    checked += 1
                assert history == problem['imitation_history_depths']
            finally:
                pool.release()
    report = dict(backend='tah_engine', verified_rows=checked, verified_problems=len(problems),
                  fixed_history_replay='passed', imitation_action_losses='passed',
                  depth1_hidden_features='bit_exact_at_storage_dtype',
                  note='Unbranched native forward_chunk replay; counterfactual isolation also tested on tiny CPU models.')
    args.cache.with_suffix('.verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def generate(args):
    from tah.engine import TaHEngine, Request, SamplingParams
    from transformers import AutoTokenizer
    checkpoint = torch.load(args.router, map_location='cpu', weights_only=True)
    meta = checkpoint['cache_metadata']
    if meta['model'] != args.model or meta.get('checkpoint_sha256', model_hashes(args.model)) != model_hashes(args.model):
        raise ValueError('Router/checkpoint model mismatch')
    router = BanditRouter(checkpoint['feature_dim'], checkpoint['width'])
    router.load_state_dict(checkpoint['state_dict'])
    router = router.to(args.device).eval().requires_grad_(False)
    threshold = float(checkpoint['report']['entropy_threshold']) if args.policy == 'entropy' else checkpoint['threshold']

    def factory(model):
        native = model.eval_iter_decider or model.iter_decider
        native.threshold.data.fill_(meta['imitation_threshold'])
        return EngineRoutingDecider(native, meta['hidden_layers'], args.policy, router, threshold,
                                    checkpoint.get('logit_threshold'))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template([dict(role='user', content=args.prompt)], tokenize=True, add_generation_prompt=True)
    with TaHEngine(args.model, device=args.device, max_batch_size=1, graph_batch_sizes=[1],
                   host_swap_bytes=0, overlap=False, decider_factory=factory) as engine:
        started = time.perf_counter()
        result = engine.generate([Request(prompt, args.max_tokens, args.seed)], SamplingParams(temperature=0))[0]
        if engine.device.type == 'cuda':
            torch.cuda.synchronize()
        report = dict(backend='tah_engine', policy=args.policy, prompt=args.prompt, output=result.text,
            token_ids=result.token_ids, depths=result.generated_pass_counts, prompt_depths=result.prompt_pass_counts,
            finish_reason=result.finish_reason, seconds=time.perf_counter() - started,
            prefill_policy=args.policy, engine_run=engine.last_run, source_sha256=source_hashes(),
            router_sha256=hashlib.sha256(args.router.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('output', 'token_ids', 'depths', 'prompt_depths', 'source_sha256')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect')
    c.add_argument('--references', type=Path, required=True)
    c.add_argument('--output-dir', type=Path, required=True)
    c.add_argument('--max-reference-tokens', type=int, default=1024)
    v = commands.add_parser('verify')
    v.add_argument('--cache', type=Path, required=True)
    v.add_argument('--max-problems', type=int, default=3)
    g = commands.add_parser('generate')
    g.add_argument('--router', type=Path, required=True)
    g.add_argument('--output', type=Path, required=True)
    g.add_argument('--policy', choices=['stop', 'continue', 'imitation', 'entropy', 'bandit'], default='bandit')
    g.add_argument('--prompt', default='What is 17 + 25?')
    g.add_argument('--max-tokens', type=int, default=512)
    for command in (c, v, g):
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--model', default='models/TaH-plus-0.6B')
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--imitation-threshold', type=float, default=0.9)
    args = p.parse_args()
    if getattr(args, 'max_reference_tokens', 1) < 1 or getattr(args, 'max_tokens', 1) < 1:
        p.error('Token limits must be positive')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
