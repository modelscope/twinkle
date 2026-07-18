import os
from pathlib import Path

from peft import LoraConfig
from tqdm import tqdm
from torch.optim import Muon  # PyTorch 2.9+; matrix-orthogonalized momentum optimizer.

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.cli import CLI
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.utils.framework import Torch
from twinkle.kernel import kernelize, liger_builtin, npu_builtin

logger = get_logger()
args = CLI.from_args()

device_mesh = DeviceMesh.from_sizes(fsdp_size=args.infra.fsdp_size, dp_size=args.infra.dp_size)
twinkle.initialize(mode=args.infra.mode, global_device_mesh=device_mesh)


def build_dataset(num_samples: int) -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset.dataset_id, data_slice=range(num_samples)))
    dataset.set_template(args.template.template_cls, model_id=args.model.model_id,
                        max_length=args.template.max_length,
                        truncation_strategy=args.template.truncation_strategy)
    dataset.map(SelfCognitionProcessor(
        args.extra.get('model_name', 'twinkle大模型'),
        args.extra.get('model_author', 'ModelScope社区'),
    ))
    dataset.encode()
    return dataset


def save_checkpoint(model: TransformersModel, checkpoint_name: str, dataloader: DataLoader):
    model.save(
        checkpoint_name,
        output_dir=args.training.output_dir,
        adapter_name=args.lora.adapter_name,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def evaluate(model):
    eval_samples = args.training.eval_samples or 100
    dataloader = DataLoader(dataset=build_dataset(eval_samples), batch_size=args.training.batch_size)
    for batch in tqdm(dataloader):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    return model.calculate_metric(is_training=False)


def train():
    train_samples = args.training.train_samples or 1000
    dataset = build_dataset(train_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=args.training.batch_size)
    model = TransformersModel(model_id=args.model.model_id)
    # Discover the actual decoder-layer class name(s) from the live model — the
    # ``model_type.title() + 'DecoderLayer'`` heuristic misnames MoE models
    # (e.g. ``qwen3_5_moe`` -> ``Qwen3_5_MoeDecoderLayer`` vs the real
    # ``Qwen3_5MoeDecoderLayer``).
    discovered = {type(m).__name__ for m in model.model.modules()
                  if type(m).__name__.endswith('DecoderLayer')}
    model.model._no_split_modules = discovered or {model.model.config.model_type.title() + 'DecoderLayer'}
    # Compose the kernel mapping: NPU built-ins first, then Liger on top so
    # `--enable-liger` opts into Liger's cross-device Triton/Ascend kernels
    # (later keys win on overlap — see twinkle.kernel Kernel.md).
    kernel_mapping = {}
    # Kernel mode: torch (TWINKLE_TORCH_BASELINE=1, no fusion) | npu (default,
    # CANN + FLA) | npu+liger (--enable-liger, Liger per-layer + fused-CE).
    _torch_baseline = os.environ.get('TWINKLE_TORCH_BASELINE', '').lower() in ('1', 'true', 'yes')
    if Torch.is_npu_available() and not _torch_baseline:
        kernel_mapping.update(npu_builtin(model))
    if args.model.enable_liger and not _torch_baseline:
        kernel_mapping.update(liger_builtin(model))
    if kernel_mapping:
        model = kernelize(model, kernel_mapping)

    # `--enable-liger` turns on BOTH the per-layer Liger/CANN kernels (above)
    # AND, by default (`enable_fused_ce=True`), the LigerFusedLinearCrossEntropyLoss
    # which skips the lm_head GEMM so the (B,T,V) logits tensor is never materialised.
    # The forward then runs under task='fused_lm_ce' (TransformersFusedCEPatch).
    # Pass `--no-fused-ce` to keep only the per-layer kernels (standard CE loss).
    # The loss is device-agnostic: on NPU/CUDA it auto-falls-back to materialised
    # CE if the fused kernel raises for a given shape (defensive).
    _use_fused_ce = args.model.enable_liger and args.model.enable_fused_ce
    _task = 'fused_lm_ce' if _use_fused_ce else 'causal_lm'

    lora_config = LoraConfig(**args.get_lora_args())
    model.add_adapter_to_model(
        args.lora.adapter_name, lora_config,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps)
    # Muon optimizes 2D hidden-layer weight matrices via Newton-Schulz orthogonalization.
    # In LoRA training the trainable params are exclusively lora_A / lora_B (both 2D),
    # so Muon applies cleanly without an AdamW fallback for 1D params.
    # ``adjust_lr_fn='match_rms_adamw'`` rescales the orthogonalized update so the same
    # lr / weight_decay tuned for AdamW can be reused directly (Moonshot Muon recipe).
    model.set_optimizer(
        optimizer_cls=Muon,
        lr=args.optimizer.learning_rate,
        adjust_lr_fn='match_rms_adamw',
    )

    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls=args.scheduler.scheduler_cls,
        num_warmup_steps=args.scheduler.num_warmup_steps,
        num_training_steps=len(dataloader))
    if _use_fused_ce:
        model.set_loss('LigerFusedLinearCrossEntropyLoss', adapter_name=args.lora.adapter_name,
                       reduction='sum')

    if args.training.resume_from_checkpoint:
        checkpoint_path = Path(args.training.resume_from_checkpoint).expanduser().resolve()
        progress = model.resume_from_checkpoint(
            str(checkpoint_path),
            resume_only_model=args.training.resume_only_model,
            adapter_name=args.lora.adapter_name)
        if not args.training.ignore_data_skip:
            dataloader.resume_from_checkpoint(progress['consumed_train_samples'])

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    optimizer_group = model.optimizer_group[args.lora.adapter_name]
    best_loss = float('inf')
    eval_interval = args.training.eval_interval or 40
    if _use_fused_ce:
        # One-time cross-rank + device drain before the first fused-CE step.
        # Under accelerate-FSDP2 the first forward lazily runs fully_shard init
        # (collectives) and the fused-CE loss issues an ad-hoc lm_head.weight
        # full_tensor() gather after the (identity) lm_head forward; if ranks
        # enter the loop slightly desynchronised (asymmetric setup), the
        # collective streams order differently across ranks and the first
        # backward deadlocks (rank A in autograd backward, rank B already in
        # the Muon optimizer's DTensor redistribution). A single barrier +
        # device sync re-aligns the ranks; later steps stay aligned because
        # the fused-CE + FSDP2 collectives are rank-symmetric. liger_bench.py
        # avoids this with a per-step _synchronize(); one pre-loop drain is
        # sufficient (verified on 35B MoE, 4x Ascend, FSDP2).
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch_npu.npu.synchronize()
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        logger.info('[fsdp2] fused-CE: pre-loop barrier+device-sync drain applied')
    for batch in dataloader:
        model.forward_backward(inputs=batch, task=_task)
        model.clip_grad_and_step()
        cur_step = optimizer_group.cur_step
        if cur_step % args.training.log_interval == 0:
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {cur_step} of {len(dataloader)}, metric: {metric}')
        if cur_step > 0 and cur_step % eval_interval == 0:
            metrics = evaluate(model)
            logger.info(f'Eval metric: {metrics}')
            metrics['step'] = cur_step
            current_loss = float(metrics['loss'])
            if current_loss < best_loss:
                save_checkpoint(model, f'checkpoint-{cur_step}', dataloader)
                best_loss = current_loss
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
