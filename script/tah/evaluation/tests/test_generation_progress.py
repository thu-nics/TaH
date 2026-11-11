"""CPU progress checks: in-flight counters, restarts, logs and artifact parity."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from queue import Queue
import threading
from types import SimpleNamespace

import pytest
import torch

from tah.evaluate import backends, jobs, progress


class Tokenizer:
    pad_token = eos_token = 'eos'

    def encode(self, text):
        return list(range(len(text))) or [1]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def test_engine_counter_inflight_restart_and_warmup(monkeypatch, tmp_path):
    """Use the real adapter; no callbacks arrive until all in-flight probes finish."""
    import tah.engine as api
    clock = [0.]
    ready, release = threading.Event(), threading.Event()
    completed, errors = [], []
    monkeypatch.setattr(backends.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(api, 'check_environment', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a: pytest.fail('GPU access'))

    def pause():
        ready.set()
        assert release.wait(5), 'test did not release fake engine'
        release.clear()

    class Engine:
        def __init__(self, path, config_override, **options):
            self.max_batch_size, self.seed = 3, 42
            self.device = torch.device('cpu')
            self.memory_budget = {}
            self.calls = 0

        @torch.inference_mode()  # frame lookup must unwrap the production decorator
        def generate(self, requests, sampling, on_result=None):
            self.calls += 1
            if on_result is None:
                # Warmup must not contribute either tokens or seconds.
                clock[0] = 100.
                self.last_run = dict(generated_tokens=999999, wasted_tokens=999, wall_seconds=100.)
                return []
            # The complete states list survives queue moves and finishes.
            states = [SimpleNamespace(generated=[1] * n, wasted_tokens=0)
                      for n in (500, 300, 200)]
            running, swapped, finished = [states[0]], [states[1]], [states[2]]
            assert running and swapped and finished
            clock[0] = 110.
            pause()
            # Expose the real engine's clear-before-wasted accounting gap.
            states[0].generated.clear()
            clock[0] = 111.
            pause()
            states[0].wasted_tokens += 500
            states[0].generated.extend([2] * 300)
            clock[0] = 120.
            pause()
            results = [api.Result(i, s.generated.copy(), 'Answer: A', [1],
                                  [1] * len(s.generated), 'length', dict(total_seconds=20.))
                       for i, s in enumerate(states)]
            for result in reversed(results):
                on_result(result)
            self.last_run = dict(generated_tokens=800, wasted_tokens=500, wall_seconds=20.)
            return results

    monkeypatch.setattr(api, 'TaHEngine', Engine)
    model, infer = backends.setup_tah_engine(dict(temperature=0., top_p=1., max_new_tokens=1000),
                                            str(tmp_path), Tokenizer())
    assert infer.progress_snapshot() is None

    def run():
        try:
            infer(['a', 'b', 'c'], on_result=lambda i, r: completed.append(i))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    rates = progress.TokenRates()
    try:
        for expected, postfix in [((10., 1000), '100 tok/s (60s), 100 avg'),
                                  ((11., 1000), '91 tok/s (60s), 91 avg'),
                                  ((20., 1300), '65 tok/s (60s), 65 avg')]:
            assert ready.wait(5), errors
            ready.clear()
            snapshot = infer.progress_snapshot()
            assert snapshot == expected
            assert rates.postfix(snapshot) == postfix
            assert completed == []
            release.set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert model.calls == 2 and completed == [2, 1, 0]
    assert infer.progress_snapshot() == (20., 1300)
    clock[0] = 1000.  # grading tail cannot dilute the final average
    assert infer.progress_snapshot() == (20., 1300)
    assert not torch.cuda.is_initialized()


def test_recent_window_and_final_snapshot():
    rates = progress.TokenRates()
    assert rates.postfix(None) == ''
    assert rates.postfix((10., 1000)) == '100 tok/s (60s), 100 avg'
    assert rates.postfix((60., 6000)) == '100 tok/s (60s), 100 avg'
    assert rates.postfix((90., 12000)) == '150 tok/s (60s), 133 avg'
    # The 60s cutoff (35s) falls between observed points; interpolation is exact
    # for this piecewise-constant fake rate.
    assert rates.postfix((95., 13000)) == '158 tok/s (60s), 137 avg'
    assert rates.postfix((95., 13000)) == '158 tok/s (60s), 137 avg'
    assert len(rates.points) == 4


def test_restart_between_counter_reads_never_double_counts():
    counter = backends._EngineTokenCounter(SimpleNamespace(generate=lambda: None))
    state = SimpleNamespace(wasted_tokens=0)

    class RestartOnLength(list):
        def __len__(self):
            length = super().__len__()
            # Emulate a thread switch after observing the old list length and
            # before reading waste: this catches the unsafe length-before-waste order.
            self.clear()
            state.wasted_tokens += length
            return length

    state.generated = RestartOnLength([1] * 1000)
    counter.states = [state]
    assert counter.snapshot()[1] == 1000
    assert state.wasted_tokens == 1000
    assert counter.snapshot()[1] == 1000
    state.generated = [2] * 400
    assert counter.snapshot()[1] == 1400


def test_log_interval_inflight_and_final(monkeypatch):
    log = io.StringIO()
    clock = [0.]
    monkeypatch.setattr(progress.sys, 'stdout', log)
    with progress.SampleProgress(240, 0, lambda: (clock[0], 100 * clock[0])) as bar:
        # Inject tqdm's clock to avoid a real ten-second test sleep.
        bar.bar._time = lambda: clock[0]
        bar.bar.start_t = bar.bar.last_print_t = 0.
        for second in range(1, 31):
            clock[0] = float(second)
            bar._refresh()
        frames = log.getvalue().split('\r')
        assert len(frames) == 5  # empty prefix + initial + 10,20,30s
        assert '0/240' in frames[-1] and '100 tok/s (60s), 100 avg' in frames[-1]
        bar.advance()
        clock[0] = 31.
    assert bar.bar.n == 1 and not bar.thread.is_alive()
    assert log.getvalue().endswith('\n')
    assert '1/240' in log.getvalue().split('\r')[-1]
    assert 'sample/s' in log.getvalue()
    assert '\x1b' not in log.getvalue()  # redirected logs have no cursor-up escapes


def test_callback_does_not_wait_for_log_writer(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    class BlockedLog(io.StringIO):
        def write(self, text):
            if threading.current_thread().name == 'eval-progress':
                entered.set()
                assert release.wait(5)
            return super().write(text)
    monkeypatch.setattr(progress.sys, 'stdout', BlockedLog())
    with progress.SampleProgress(2, 0, interval=.01) as bar:
        bar.bar.mininterval = 0.
        try:
            assert entered.wait(5)
            # This callback-side operation must finish while logging is blocked.
            bar.advance()
            assert bar.done == 1
        finally:
            release.set()
    assert bar.bar.n == 1 and not bar.thread.is_alive()


def test_backend_bar_updates_before_first_finished_sample(monkeypatch, tmp_path, capsys):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a: Tokenizer())
    observed = Queue()
    counter = [None]

    class FastProgress(progress.SampleProgress):
        def __init__(self, *args):
            super().__init__(*args, interval=.01)
            self.bar.mininterval = .01

        def _refresh(self):
            super()._refresh()
            observed.put((self.bar.n, self.bar.postfix))

    monkeypatch.setattr(jobs, 'SampleProgress', FastProgress)

    def infer(prompts, on_result):
        for snapshot, expected in [((10., 1000), '100 tok/s (60s), 100 avg'),
                                    ((20., 3000), '150 tok/s (60s), 150 avg')]:
            counter[0] = snapshot
            while True:
                done, postfix = observed.get(timeout=5)
                if postfix == expected:
                    assert done == 0
                    break
        for i in (1, 0):
            on_result(i, ('Answer: A', .2, [1, 2]))

    infer.supports_callback = True
    infer.progress_snapshot = lambda: counter[0]
    monkeypatch.setattr(jobs, 'setup_backend', lambda *a, **kw: (None, infer))
    jobs.run_single_job(config=dict(batch_size=2, repeat_size=1, grading_workers=0),
        combined_dataset_name='gpqa', dataset_names=['gpqa'], output_dir=str(tmp_path),
        timestamp='test', model_path='stub', job_id=0, job_nums=1, start_idx=0,
        end_idx=2, tp_size=1, backend='tah_engine', data_range=None,
        problems_data=[dict(id=f'gpqa_{i}', question='question', answer='A') for i in range(2)],
        field_mapping=dict(id_field='id', question_field='question', answer_field='answer', answer_type='boxed'))
    stdout, stderr = capsys.readouterr()
    assert stderr == ''
    assert '0/2' in stdout and '150 tok/s (60s), 150 avg' in stdout
    assert '2/2' in stdout


@pytest.mark.parametrize('callback', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_harness_bar_and_error_cleanup(monkeypatch, tmp_path, capsys, callback, fail):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a: Tokenizer())
    cleaned = []
    monkeypatch.setattr(jobs, 'cleanup', lambda *a: cleaned.append(True))
    result = dict(text='Answer: A', seconds=.2, pass_counts=[1, 2], token_ids=[3, 4],
                  prompt_pass_counts=[1], finish_reason='length', timing={})
    if callback:
        def infer(prompts, on_result):
            on_result(1, result)
            if fail:
                raise RuntimeError('fake failure')
            on_result(0, result)
        infer.supports_callback = True
        infer.progress_snapshot = lambda: (10., 1000)
    else:
        def infer(prompts):
            if fail:
                raise RuntimeError('fake failure')
            return [result] * len(prompts)
    monkeypatch.setattr(jobs, 'setup_backend', lambda *a, **kw: (None, infer))
    kwargs = dict(config=dict(batch_size=2, repeat_size=1, grading_workers=0),
        combined_dataset_name='gpqa', dataset_names=['gpqa'], output_dir=str(tmp_path),
        timestamp='test', model_path='stub', job_id=0, job_nums=1, start_idx=0,
        end_idx=2, tp_size=1, backend='tah_engine' if callback else 'tah', data_range=None,
        problems_data=[dict(id=f'gpqa_{i}', question='question', answer='A') for i in range(2)],
        field_mapping=dict(id_field='id', question_field='question', answer_field='answer',
                           answer_type='boxed'))
    if fail:
        with pytest.raises(RuntimeError, match='fake failure'):
            jobs.run_single_job(**kwargs)
    else:
        jobs.run_single_job(**kwargs)
    stdout, stderr = capsys.readouterr()
    assert stderr == '' and 'Job 0:' in stdout
    assert 'batches' not in stdout and 'finished samples only' not in stdout
    assert ('tok/s' in stdout) == callback
    final = stdout.split('\r')[-1].split('\n')[0]
    assert f'{1 if callback and fail else 0 if fail else 2}/2' in final
    assert cleaned == [True]
    assert not any(t.name == 'eval-progress' for t in threading.enumerate())
    if not fail:
        job_dir = next(tmp_path.rglob('job_0'))
        jobs.combine_job_results(job_dir.parent, 1)
        # Fixed data expectations also guard callback-order CSV serialization.
        rows = list(csv.DictReader((job_dir / 'detailed_results.csv').open()))
        assert [r['problem_id'] for r in rows] == (['gpqa_1', 'gpqa_0'] if callback else ['gpqa_0', 'gpqa_1'])
        assert all(r['is_correct'] == 'True' for r in rows)
        assert len((job_dir.parent / 'samples.jsonl').read_text().splitlines()) == 2


def test_prechange_artifacts_byte_identical(monkeypatch, tmp_path):
    """Pin pre-change bytes; also replay the captured harness during local audit."""
    before = Path('data/tmp/wp20/jobs.py.before')
    modules = []
    if before.exists():
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader('wp20_before_jobs', SourceFileLoader('wp20_before_jobs', str(before)))
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        modules.append((baseline, tmp_path / 'before'))
    modules.append((jobs, tmp_path / 'after'))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a: Tokenizer())
    def infer(prompts, on_result):
        for i in (3, 1, 2, 0):
            on_result(i, dict(text='Answer: A', seconds=.2, pass_counts=[1, 2],
                             token_ids=[3, 4], prompt_pass_counts=[1], finish_reason='length', timing={}))
    infer.supports_callback = True
    infer.progress_snapshot = lambda: (10., 1000)
    artifacts = []
    for module, directory in modules:
        monkeypatch.setattr(module, 'setup_backend', lambda *a, **kw: (None, infer))
        monkeypatch.setattr(module, 'cleanup', lambda *a, **kw: None)
        module.run_single_job(config=dict(batch_size=2, repeat_size=2, grading_workers=0),
            combined_dataset_name='gpqa', dataset_names=['gpqa'], output_dir=str(directory),
            timestamp='test', model_path='stub', job_id=0, job_nums=1, start_idx=0,
            end_idx=2, tp_size=1, backend='tah_engine', data_range=None,
            problems_data=[dict(id=f'gpqa_{i}', question='question', answer='A') for i in range(2)],
            field_mapping=dict(id_field='id', question_field='question', answer_field='answer', answer_type='boxed'))
        run_dir = next(directory.rglob('job_0')).parent
        module.combine_job_results(run_dir, 1)
        artifacts.append({str(p.relative_to(run_dir)): p.read_bytes()
                          for p in run_dir.rglob('*') if p.is_file() and
                          (p.suffix in ('.csv', '.jsonl') or p.name.startswith('sample_'))})
    assert artifacts[0] == artifacts[-1]
    # SHA256 captured from the actual pre-WP20 harness, not from the new path.
    assert {name: hashlib.sha256(data).hexdigest() for name, data in artifacts[-1].items()} == {
        'detailed_results.csv': '2eb77102e6150ffe0db3bc2310ac7f7b2f2698f795dba4bdf6be1fa89685f91a',
        'evaluation_stats.csv': '581125b22e6956e5094bc515d8877c61a0788072a2a734445413cb1113b54334',
        'job_0/detailed_results.csv': '444dfe1324cbc95ce36f2d3b0f2b31a2e367648fe6647204c8e37f839684e1f9',
        'job_0/details/gpqa_0/sample_0.json': '20fa97d623a3458023cfe5dbc4b48f82c61da55a4a7232a90503c7bc5925b03f',
        'job_0/details/gpqa_0/sample_1.json': '20fa97d623a3458023cfe5dbc4b48f82c61da55a4a7232a90503c7bc5925b03f',
        'job_0/details/gpqa_1/sample_0.json': 'afcb5d96375174f88da245214076d2c3679a0b2480ac6ff42b4005581f624dc6',
        'job_0/details/gpqa_1/sample_1.json': 'afcb5d96375174f88da245214076d2c3679a0b2480ac6ff42b4005581f624dc6',
        'job_0/evaluation_stats.csv': 'd0e1c811aa02acabd8aaaedf750f30c9b6e33834618e7fa2eddb48e44c64b53a',
        'samples.jsonl': '43ea194ce311d9dc9a07c136e15a5003aea30e1477d58630ada801cd21cb13a8',
    }
    assert all(json.loads(line)['sample'] in (0, 1)
               for line in artifacts[-1]['samples.jsonl'].splitlines())
