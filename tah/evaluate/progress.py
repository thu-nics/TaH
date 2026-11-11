"""Log-friendly sample progress; rendering stays off the generation thread."""
from collections import deque
import sys
import threading

from tqdm import tqdm


class TokenRates:
    """Wall-clock gross token rates, with a rolling 60-second window."""

    def __init__(self, window=60.):
        self.window = window
        self.points = deque([(0., 0)])

    def postfix(self, snapshot):
        if snapshot is None:
            return ''
        elapsed, total = snapshot
        if elapsed > self.points[-1][0]:
            self.points.append((elapsed, total))
        # Keep one point before the cutoff for linear interpolation at exactly 60s.
        cutoff = max(0., elapsed - self.window)
        while len(self.points) > 2 and self.points[1][0] <= cutoff:
            self.points.popleft()
        start, tokens = self.points[0]
        if len(self.points) > 1 and start < cutoff:
            end, end_tokens = self.points[1]
            tokens += (end_tokens - tokens) * (cutoff - start) / (end - start)
            start = cutoff
        recent = (total - tokens) / max(elapsed - start, 1e-9)
        average = total / max(elapsed, 1e-9)
        return f'{recent:.0f} tok/s (60s), {average:.0f} avg'


class SampleProgress:
    """Callbacks only increment an int; one daemon renders every five seconds.

    Neither the engine nor its callback takes a rendering lock or writes the bar.
    Joining and the forced final refresh happen after inference, including errors.
    """

    def __init__(self, total, job_id, snapshot=None, interval=5.):
        self.done = 0
        self.snapshot = snapshot
        self.interval = interval
        self.rates = TokenRates()
        self.stop = threading.Event()
        self.bar = tqdm(total=total, desc=f'Job {job_id}', unit='sample',
                        file=sys.stdout, mininterval=10., miniters=0, leave=True,
                        dynamic_ncols=False, ascii=True, disable=False)
        self.thread = threading.Thread(target=self._monitor, name='eval-progress', daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def advance(self):
        self.done += 1

    def _refresh(self):
        if self.snapshot is not None:
            self.bar.set_postfix_str(self.rates.postfix(self.snapshot()), refresh=False)
        self.bar.update(self.done - self.bar.n)

    def _monitor(self):
        while not self.stop.wait(self.interval):
            self._refresh()

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        try:
            self._refresh()
        finally:
            self.bar.close()  # tqdm always writes the final state and a newline
