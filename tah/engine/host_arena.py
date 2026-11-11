"""Fixed host arena for request swapping: exactly the budget, allocated once."""
import weakref

import torch


def _unregister(buffer):
    # Teardown only: wait for copies into the arena, then unpin before the
    # buffer is freed (the finalizer holds the last reference to it).
    try:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
    except Exception:  # interpreter shutdown or a lost CUDA context
        pass


class HostArena:
    """``nbytes`` of host memory for swapped state; page-locked on CUDA.

    ``torch.empty(pin_memory=True)`` uses PyTorch's caching host allocator,
    which rounds each block up to a power of two (a 10 GiB budget would pin
    16 GiB) and keeps freed blocks cached. Instead a plain host buffer of
    exactly ``nbytes`` is registered with ``cudaHostRegister``, which pins it in
    place: the pinned size is the budget, from startup to teardown. On CPU the
    buffer is an ordinary tensor, so the paging logic is the same in CPU tests.

    ``close()`` releases it deterministically (idempotent); otherwise a finalizer
    releases it when the arena object is collected.
    """
    def __init__(self, nbytes, device):
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError('host_swap_bytes must be a nonnegative integer')
        self.nbytes, self.device = nbytes, torch.device(device)
        self.pinned = self.device.type == 'cuda' and nbytes > 0
        self.buffer = self._finalizer = None
        self.closed = False
        if not nbytes:
            return
        message = (f'cannot reserve the host swap arena: host_swap_bytes={nbytes} '
                   f'({nbytes / 1024**3:.2f} GiB) of {"pinned " if self.pinned else ""}host RAM; '
                   'lower host_swap_bytes or free host memory')
        try:
            self.buffer = torch.empty(nbytes, dtype=torch.uint8)
        except RuntimeError as error:
            raise RuntimeError(message) from error
        if self.pinned:
            cudart = torch.cuda.cudart()
            status = cudart.cudaHostRegister(self.buffer.data_ptr(), nbytes, 0)
            if status != cudart.cudaError.success:
                self.buffer = None
                raise RuntimeError(f'{message} (cudaHostRegister: {cudart.cudaGetErrorString(status)})')
            self._finalizer = weakref.finalize(self, _unregister, self.buffer)
            if not self.buffer.is_pinned():
                raise RuntimeError(f'{message} (registered memory is not reported as pinned)')

    def close(self):
        """Unpin (after a device sync) and free the buffer now; safe to repeat."""
        if self._finalizer is not None:
            self._finalizer()  # runs at most once; then drops its buffer reference
        self.buffer, self.closed = None, True

    def view(self, offset, nbytes):
        """Byte view of the arena; no allocation."""
        return self.buffer[offset:offset + nbytes]
