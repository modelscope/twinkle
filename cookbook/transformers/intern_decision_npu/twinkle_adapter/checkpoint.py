"""FSDP2 sharded optimizer I/O via PyTorch DCP; HF model export stays separate."""
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict,
)


def reshard_for_checkpoint(model):
    """Restore sharded parameter identities after inference without backward."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def install_sharded_optimizer_io(strategy, cpu_group, fence):
    def save(model, optimizer, output_path, *, param_name_mapping=None):
        if param_name_mapping:
            raise ValueError('Only full-parameter optimizer names are supported')
        inner = getattr(optimizer, 'optimizer', optimizer)
        reshard_for_checkpoint(model)
        torch.npu.synchronize()
        fence()
        state = {'optimizer': get_optimizer_state_dict(
            model, inner, options=StateDictOptions(full_state_dict=False, cpu_offload=False))}
        dcp.save(state, checkpoint_id=output_path, process_group=cpu_group)
        fence()

    def load(model, optimizer, input_path, *, param_name_mapping=None):
        if param_name_mapping:
            raise ValueError('Only full-parameter optimizer names are supported')
        if not (Path(input_path) / '.metadata').is_file():
            raise ValueError('Missing completed distributed optimizer checkpoint')
        inner = getattr(optimizer, 'optimizer', optimizer)
        reshard_for_checkpoint(model)
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        state = {'optimizer': get_optimizer_state_dict(model, inner, options=options)}
        dcp.load(state, checkpoint_id=input_path, process_group=cpu_group)
        set_optimizer_state_dict(model, inner, state['optimizer'], options=options)
        torch.npu.synchronize()
        fence()

    strategy.save_optimizer_checkpoint = save
    strategy.load_optimizer_checkpoint = load
