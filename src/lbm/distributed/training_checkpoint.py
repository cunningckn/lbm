"""Complete, atomically published distributed training checkpoints (fixed world size)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

from lbm.checkpoint import capture_rng_state


def _rank_zero(operation):
    result = [None]
    if dist.get_rank() == 0:
        try:
            operation()
        except Exception as exc:
            result[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(result)
    if result[0]:
        raise RuntimeError(result[0])


def save_training_checkpoint(path, model, optimizer, scheduler, *, step, epoch, batch_in_epoch,
                             epoch_rng, signature):
    path = Path(path)
    temporary = path.with_name(path.name + '.incomplete')
    rank = dist.get_rank()
    _rank_zero(lambda: temporary.mkdir(parents=True, exist_ok=False))
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict() if signature['fsdp'] else get_optimizer_state_dict(model, optimizer)}
    dcp.save(state, checkpoint_id=str(temporary / 'shards'))
    torch.save(dict(step=step, epoch=epoch, batch_in_epoch=batch_in_epoch, epoch_rng=epoch_rng,
                    rng_state=capture_rng_state(), scheduler=scheduler.state_dict(), signature=signature),
               temporary / f'rank-{rank}.pt')
    dist.barrier()
    def publish():
        (temporary / 'complete.json').write_text(json.dumps({'version': 1, 'world': dist.get_world_size()}))
        temporary.rename(path)
    _rank_zero(publish)


def load_training_checkpoint(path, model, optimizer, scheduler, *, signature):
    path = Path(path)
    manifest = json.loads((path / 'complete.json').read_text())
    if manifest != {'version': 1, 'world': dist.get_world_size()}:
        raise ValueError('distributed checkpoint requires the same world size and format')
    payload = torch.load(path / f'rank-{dist.get_rank()}.pt', map_location='cpu', weights_only=False)
    if payload['signature'] != signature:
        raise ValueError('Resume configuration or data identity differs from distributed checkpoint')
    step, epoch, cursor = (payload[k] for k in ('step', 'epoch', 'batch_in_epoch'))
    batches = signature['batches_per_epoch']
    if any(type(v) is not int or v < 0 for v in (step, epoch, cursor)) or cursor > batches:
        raise ValueError('invalid distributed checkpoint cursor')
    if step != epoch * batches + cursor:
        raise ValueError('invalid distributed checkpoint step')
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict() if signature['fsdp'] else get_optimizer_state_dict(model, optimizer)}
    dcp.load(state, checkpoint_id=str(path / 'shards'))
    model.load_state_dict(state['model'])
    if signature['fsdp']:
        optimizer.load_state_dict(state['optimizer'])
    else:
        set_optimizer_state_dict(model, optimizer, state['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    return payload
