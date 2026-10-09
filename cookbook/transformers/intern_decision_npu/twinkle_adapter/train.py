"""Full language-parameter decision SFT with Twinkle; initial NPU smoke entry."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, get_cosine_with_min_lr_schedule_with_warmup
import twinkle
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.kernel.ops.fla.npu import apply_qwen3_5_fla
from .data import count_supervised_tokens, encode_decision
from .processor import DecisionProcessor
from .checkpoint import install_sharded_optimizer_io, reshard_for_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/models/Qwen3.5-4B')
    parser.add_argument('--data', default='/data/joint/train.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--schedule-steps', type=int, default=120)
    parser.add_argument('--model-only', action='store_true', help='Final weights only; no optimizer resume')
    parser.add_argument('--global-batch', type=int, default=16)
    parser.add_argument('--save-every', type=int, default=150)
    parser.add_argument('--resume')
    parser.add_argument('--validation')
    args = parser.parse_args()
    world = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 4 or args.global_batch % world:
        raise ValueError('This entry was configured for four NPU workers')
    if args.steps > args.schedule_steps:
        raise ValueError('Requested steps exceed the fixed LR schedule')
    torch.npu.set_device(local_rank)
    mesh = DeviceMesh.from_sizes(fsdp_size=world, dp_size=1)
    twinkle.initialize(mode='local', nproc_per_node=world, global_device_mesh=mesh, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.resume or args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<decision>']})
    records = [json.loads(s) for s in Path(args.data).read_text().splitlines()]
    features = [encode_decision(tokenizer, r) for r in records]
    if len(features) % args.global_batch:
        raise ValueError('Use complete global training batches for this baseline')
    steps_per_epoch = len(features) // args.global_batch
    # An explicit deterministic permutation for each epoch is shared by all ranks;
    # Twinkle's sampler then slices each GLOBAL batch exactly once across the mesh.
    generator = torch.Generator().manual_seed(42)
    ordered = []
    for _ in range(math.ceil(args.schedule_steps / steps_per_epoch)):
        ordered.extend(features[i] for i in torch.randperm(len(features), generator=generator).tolist())
    dataset = Dataset(dataset_meta=DatasetMeta(data=ordered))
    loader = DataLoader(dataset=dataset, device_mesh=mesh, batch_size=args.global_batch,
                        shuffle=False, num_workers=0, drop_last=True)
    validation_features = None
    if args.validation:
        validation_features = [encode_decision(tokenizer, json.loads(s))
                               for s in Path(args.validation).read_text().splitlines()]
    apply_qwen3_5_fla()
    model = TransformersModel(
        model_id=args.resume or args.model, device_mesh=mesh,
        dtype=torch.float32, attn_implementation='sdpa', mixed_precision='bf16',
        memory_efficient_init=True, strategy='accelerate',
        fsdp_config={'reshard_after_forward': True, 'activation_checkpointing': True,
                     'cpu_ram_efficient_loading': True, 'state_dict_type': 'FULL_STATE_DICT'})
    model.model.gradient_checkpointing_disable()
    model.model.config.use_cache = False
    model.model.config.text_config.use_cache = False
    apply_qwen3_5_fla(model)
    model.model._no_split_modules = sorted({type(m).__name__ for m in model.model.modules()
                                            if type(m).__name__.endswith('DecoderLayer')})
    frozen = 0
    trainable = 0
    for name, param in model.model.named_parameters():
        is_vision = name.startswith('visual.') or '.visual.' in name
        param.requires_grad_(not is_vision)
        if is_vision:
            frozen += param.numel()
        else:
            trainable += param.numel()
    if frozen == 0 or trainable == 0:
        raise ValueError('Vision/language parameter partition failed')
    if max(tokenizer.encode('<decision>', add_special_tokens=False)) >= model.model.get_input_embeddings().num_embeddings:
        raise ValueError('Marker is outside the model vocabulary')
    group = model.optimizer_group['']
    # Twinkle registers a forward hook even for pre-encoded text batches.
    # Retain its template; input_ids bypass batch_encode, so labels are not shifted again.
    if hasattr(group.template.processor, 'tokenizer'):
        group.template.processor.tokenizer = tokenizer
    else:
        group.template.processor = tokenizer
    model._default_tokenizer = tokenizer
    model.set_processor(DecisionProcessor, pad_token_id=tokenizer.pad_token_id, pad_multiple=128)
    model.set_loss('CrossEntropyLoss', reduction='sum')
    model.set_optimizer('AdamW', lr=2e-6, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    group.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        group.optimizer, num_warmup_steps=math.ceil(args.schedule_steps * 0.03),
        num_training_steps=args.schedule_steps, min_lr=2e-7)
    # Prevent early HCCL collectives from overlapping rank-zero CPU offload.
    cpu_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=20))
    def fence():
        dist.monitored_barrier(group=cpu_group, timeout=timedelta(minutes=20))
    for method in ['get_full_state_dict']:
        original = getattr(model.strategy, method)
        def guarded(*a, _original=original, **kw):
            reshard_for_checkpoint(model.model)
            result = _original(*a, **kw)
            # Accelerate activation checkpointing introduces a wrapper namespace.
            # Export the original HF names so loading cannot silently miss layers.
            normalized = {}
            for key, value in result.items():
                canonical = key.replace('._checkpoint_wrapped_module.', '.')
                if canonical in normalized:
                    raise RuntimeError(f'Duplicate export key: {canonical}')
                normalized[canonical] = value
            fence()
            return normalized
        setattr(model.strategy, method, guarded)
    install_sharded_optimizer_io(model.strategy, cpu_group, fence)
    if args.resume:
        progress = model.resume_from_checkpoint(args.resume)
        loader.resume_from_checkpoint(progress['consumed_train_samples'])
        model._load_rng_state(str(Path(args.resume) / f'rng_state_rank{rank}.pt'))
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    fence()
    best_path = output / 'best-selection.json'
    best = json.loads(best_path.read_text()) if best_path.exists() else {'loss': float('inf')}

    def evaluate():
        validation_loader = DataLoader(
            dataset=Dataset(dataset_meta=DatasetMeta(data=validation_features)),
            device_mesh=mesh, batch_size=args.global_batch, shuffle=False,
            num_workers=0, drop_last=False)
        loss_sum, decisions, cases = 0.0, 0, 0
        for validation_batch in validation_loader:
            decisions += count_supervised_tokens(validation_batch)
            cases += len(validation_batch)
            model.forward_only(inputs=validation_batch)
            loss_sum += model.calculate_loss()
        model.calculate_metric(is_training=False)
        totals = torch.tensor([loss_sum, decisions, cases], dtype=torch.float64)
        dist.all_reduce(totals, group=cpu_group)
        expected_decisions = count_supervised_tokens(validation_features)
        if int(totals[1]) != expected_decisions or int(totals[2]) != len(validation_features):
            raise RuntimeError('Validation sampler did not cover each decision exactly once')
        loss = float(totals[0] / totals[1])
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite validation loss')
        return loss
    if rank == 0:
        (output / 'run-config.json').write_text(json.dumps({
            **vars(args), 'trainable_parameters': trainable, 'frozen_parameters': frozen,
            'twinkle_revision': 'eff13b93a8576f773f11c85adccffdd3eb22fab3',
            'status': 'running', 'precision_acceptance': 'not_yet_evaluated'}, indent=2))
    started = time.monotonic()
    for batch in loader:
        result = model.forward_backward(inputs=batch)
        if not math.isfinite(float(result['loss'])):
            raise RuntimeError('Non-finite training loss')
        model.clip_grad_and_step(max_grad_norm=1.0)
        if not math.isfinite(float(group._last_grad_norm)):
            raise RuntimeError('Non-finite gradient norm; stopping before checkpoint save')
        step = group.cur_step
        metrics = model.calculate_metric(is_training=True)
        if rank == 0:
            record = {'step': step, 'elapsed_s': time.monotonic() - started, 'metrics': metrics,
                      'peak_memory_gib': torch.npu.max_memory_allocated() / 1024**3}
            with (output / 'logging.jsonl').open('a') as handle:
                handle.write(json.dumps(record, default=str) + '\n')
            print(json.dumps(record, default=str), flush=True)
        if step % args.save_every == 0 or step >= args.steps:
            fence()
            name = f'checkpoint-{step}'
            model.save(name, output_dir=str(output), save_optimizer=not args.model_only,
                       consumed_train_samples=loader.get_state()['consumed_train_samples'])
            fence()
            validation_loss = evaluate() if validation_features else None
            torch.save(model._get_training_rng_state(), output / name / f'rng_state_rank{rank}.pt')
            fence()
            if rank == 0:
                for config in Path(args.model).glob('*processor*.json'):
                    shutil.copy2(config, output / name / config.name)
                if validation_loss is not None:
                    with (output / 'validation.jsonl').open('a') as handle:
                        handle.write(json.dumps({'step': step, 'loss': validation_loss}) + '\n')
                    if validation_loss < best['loss']:
                        best = {'loss': validation_loss, 'step': step,
                                'checkpoint': str(output / name)}
                        best_path.write_text(json.dumps(best, indent=2))
                    # Retain the latest recoverable checkpoint plus validation-best.
                    keep = {str(output / name), best.get('checkpoint')}
                    for old in output.glob('checkpoint-*'):
                        if old.is_dir() and str(old) not in keep:
                            shutil.rmtree(old)
            fence()
        if step >= args.steps:
            break
    if group.cur_step != args.steps:
        raise RuntimeError(f'Incomplete training: {group.cur_step} steps, expected {args.steps}')
    if rank == 0:
        (output / 'training-complete.json').write_text(json.dumps({'step': group.cur_step,
            'checkpoint': str(output / f'checkpoint-{group.cur_step}'),
            'best_model_checkpoint': best.get('checkpoint'),
            'accuracy_evaluated': False, 'optimizer_saved': not args.model_only}, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
