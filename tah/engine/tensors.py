"""Small, asynchronous host descriptor uploads for eager scheduling."""
import torch


def upload(values, device, dtype=torch.long):
    device = torch.device(device)
    host = torch.tensor(values, dtype=dtype, pin_memory=device.type == 'cuda')
    return host.to(device=device, non_blocking=True)
