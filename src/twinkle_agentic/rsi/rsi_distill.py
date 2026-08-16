# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 4 — turn the data collected by ``llm_backup`` into a per-role LoRA
via plain SFT.

Where the data comes from
-------------------------
``twinkle_agentic.utils.llm_backup`` now optionally appends one raw record per
teacher call to the JSONL at ``$LLM_BACKUP_DUMP_PATH`` (off unless that env var
is set). Each line is::

    {"key": <hash>, "trajectory": <the exact model input>,
     "student": <student output str>, "teacher": <teacher output str>,
     "match": <bool>}

``trajectory`` is whatever the decorated role passed as its ``trajectory`` arg —
in twinkle_agentic that is an ``{"messages": [...], "tools": [...]}`` dict (see
protocol/openai.py). It is stored verbatim so it can be reshaped here into an
SFT pair without re-running anything.

What this script trains
------------------------
SFT target = EVERY teacher output (the ``match`` flag is ignored; decided by the
user). One training sample is::

    messages = <trajectory messages> + [{"role": "assistant", "content": teacher}]

and only that final assistant turn is trainable (``key_rounds=[len(msgs)-1]``),
so the LoRA learns to reproduce the teacher's output for that role. Run the
script once per auxiliary role, each with its own dump file and adapter name —
that is the composable unit; there is no multi-role loop here on purpose.

Status: this is the PLUMBING. It only runs distillation when invoked explicitly
(``python -m twinkle_agentic.rsi.rsi_distill --input ...``); importing it does
nothing.

Numbers are INHERITED, not invented:
  * LR / BATCH_SIZE / MICRO_BATCH / EPOCHS / MAX_MODEL_LEN  <- e18_sft_kod.py
  * LORA_RANK / alpha=rank*2 / dropout=0.05                 <- rsi_rl.py
All are overridable via the env vars below.

Env vars
--------
    RSI_DUMP_PATH   dump JSONL to read (default $LLM_BACKUP_DUMP_PATH)
    RSI_ADAPTER     adapter name to train + save (default: derived from dump name)
    OUTPUT_DIR      where the adapter is written (default output/rsi/distill)
    MODEL_ID        base model (default Qwen/Qwen3-4B)
    plus TRAIN_GPUS / TRAIN_FSDP / GPU_MEM / SEED / EPOCHS / BATCH_SIZE /
    MICRO_BATCH / LR / MAX_MODEL_LEN / LORA_RANK (all inherited defaults).
"""
import argparse
import json
import os
import random
import shutil
import time
from typing import Any, Dict, List

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import pack_user_data
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Template

logger = get_logger()

# ── config (env, all defaults inherited from e18_sft_kod.py / rsi_rl.py) ────
MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3-4B')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join('output', 'rsi', 'distill'))

TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 4))
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', 1))
TRAIN_DP = TRAIN_GPUS // TRAIN_FSDP
GPU_MEM = float(os.environ.get('GPU_MEM', 0.8))

SEED = int(os.environ.get('SEED', 42))
EPOCHS = float(os.environ.get('EPOCHS', 1))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
MICRO_BATCH = int(os.environ.get('MICRO_BATCH', 8))
LR = float(os.environ.get('LR', 1e-5))
MAX_MODEL_LEN = int(os.environ.get('MAX_MODEL_LEN', 16000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

LOG_EVERY_STEPS = int(os.environ.get('LOG_EVERY_STEPS', 1))
RUN_ID = time.strftime('%m%d-%H%M%S')


# ===========================================================================
# data: llm_backup dump -> SFT samples (target = every teacher output)
# ===========================================================================
def _trajectory_messages(traj: Any) -> List[Dict[str, Any]]:
    """Pull the message list out of a dumped ``trajectory``.

    twinkle_agentic passes trajectory as ``{"messages": [...], "tools": ...}``;
    tolerate a bare list of messages too so the loader does not depend on one
    role's exact calling convention.
    """
    if isinstance(traj, dict):
        msgs = traj.get('messages')
    elif isinstance(traj, list):
        msgs = traj
    else:
        msgs = None
    return msgs if isinstance(msgs, list) else []


def load_samples(dump_path: str) -> List[Dict[str, Any]]:
    """Read the llm_backup dump and build one SFT sample per usable record.

    A record is usable when it has a non-empty trajectory message list AND a
    non-empty teacher string. The teacher output becomes the sole trainable
    assistant turn appended to the trajectory messages. ``match`` is ignored on
    purpose (target = every teacher output).
    """
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f'找不到 llm_backup dump：{dump_path}')
    raw, bad_json = [], 0
    with open(dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception:
                bad_json += 1
    drop = {'no_messages': 0, 'no_teacher': 0}
    out = []
    for r in raw:
        msgs = _trajectory_messages(r.get('trajectory'))
        if not msgs:
            drop['no_messages'] += 1
            continue
        teacher = r.get('teacher')
        if not isinstance(teacher, str) or not teacher.strip():
            drop['no_teacher'] += 1
            continue
        sample_msgs = list(msgs) + [{'role': 'assistant', 'content': teacher}]
        out.append({'messages': sample_msgs, 'key': r.get('key', '')})
    logger.info(f'[data] 读入 {len(raw)} 条'
                + (f'（跳过 {bad_json} 行半行）' if bad_json else '')
                + f'，可训 {len(out)} 条，丢弃明细 {drop}')
    return out


def make_trajs(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """样本 -> twinkle 轨迹。只把最后一轮（teacher 输出）标为可训区。"""
    trajs = []
    for s in batch:
        msgs = s['messages']
        trajs.append({'messages': msgs,
                      'user_data': pack_user_data({'key_rounds': [len(msgs) - 1]})})
    return trajs


# ===========================================================================
# model: base + one LoRA adapter (rsi_rl.py's config, TransformersModel SFT)
# ===========================================================================
def build_model(adapter_name: str):
    twinkle.initialize(mode='ray', nproc_per_node=TRAIN_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='train', ranks=list(range(TRAIN_GPUS)), device_type='GPU')])
    model = TransformersModel(
        model_id=MODEL_ID, remote_group='train',
        device_mesh=DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_DP,
                                          fsdp_size=TRAIN_FSDP),
        ddp_config={'find_unused_parameters': False})
    # enable_thinking=True: auxiliary roles produce reasoning before their answer,
    # same deployment mode as the teacher that generated the targets.
    model.set_template(Template, model_id=MODEL_ID, enable_thinking=True,
                       max_length=MAX_MODEL_LEN, truncation_strategy='delete')
    model.set_processor(InputProcessor, padding_free=False)
    model.set_loss('CrossEntropyLoss')
    lora_cfg = LoraConfig(target_modules='all-linear', r=LORA_RANK,
                          lora_alpha=LORA_RANK * 2, lora_dropout=0.05)
    model.add_adapter_to_model(adapter_name, lora_cfg,
                               gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LR)
    return model


def step_metrics(model) -> Dict[str, float]:
    """取本 step 的优化指标（loss/grad_norm/lr）。twinkle 把 loss 格式化成字符串，
    所以用 float() 试转而不是 isinstance 判数值型，否则会静默丢掉 loss。"""
    out: Dict[str, float] = {}
    for k, val in (model.calculate_metric(is_training=True) or {}).items():
        if isinstance(val, bool):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if k.startswith('learning rate'):
            if 'group 1' in k:
                out['lr'] = fval
        else:
            out[k.replace(' ', '_')] = fval
    return out


def archive_output_dir() -> None:
    """启动时把已存在的非空 OUTPUT_DIR 整个 mv 走（sft_log.jsonl 是追写的），
    否则重跑会把两条 loss 曲线焊进一个文件。与 e18_sft_kod 同一套机制。"""
    if not os.path.isdir(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return
    stamp = time.strftime('%m%d-%H%M%S', time.localtime(os.path.getmtime(OUTPUT_DIR)))
    dst = f'{OUTPUT_DIR}.bak-{stamp}'
    i = 1
    while os.path.exists(dst):
        dst = f'{OUTPUT_DIR}.bak-{stamp}-{i}'
        i += 1
    shutil.move(OUTPUT_DIR, dst)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f'[init] 旧输出目录已归档 -> {dst}')


# ===========================================================================
# train
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='RSI step 4: distill one auxiliary-role LoRA from an llm_backup dump.')
    parser.add_argument('--input', default=os.environ.get('RSI_DUMP_PATH', os.environ.get('LLM_BACKUP_DUMP_PATH')),
                        help='llm_backup dump JSONL (default $RSI_DUMP_PATH / $LLM_BACKUP_DUMP_PATH)')
    parser.add_argument('--adapter', default=os.environ.get('RSI_ADAPTER'),
                        help='adapter name to train and save (default: derived from dump filename)')
    args = parser.parse_args()

    dump_path = args.input
    if not dump_path:
        raise SystemExit('必须给 --input（或设 $RSI_DUMP_PATH / $LLM_BACKUP_DUMP_PATH）指向 llm_backup dump')
    adapter_name = args.adapter or os.path.splitext(os.path.basename(dump_path))[0]

    t0 = time.time()
    archive_output_dir()
    samples = load_samples(dump_path)
    if len(samples) < BATCH_SIZE:
        raise RuntimeError(f'可用样本 {len(samples)} 条 < BATCH_SIZE {BATCH_SIZE}')
    if BATCH_SIZE % TRAIN_DP:
        raise RuntimeError(f'BATCH_SIZE({BATCH_SIZE}) 必须是 TRAIN_DP({TRAIN_DP}) 的整倍数')

    model = build_model(adapter_name)
    steps_per_epoch = len(samples) // BATCH_SIZE
    total_steps = int(steps_per_epoch * EPOCHS)
    logger.info(f'RSI-DISTILL start: adapter={adapter_name} n={len(samples)} bs={BATCH_SIZE} '
                f'micro={MICRO_BATCH} lr={LR} epochs={EPOCHS} steps/epoch={steps_per_epoch} '
                f'total_steps={total_steps} rank={LORA_RANK} gpus={TRAIN_GPUS} out={OUTPUT_DIR}')

    log_path = os.path.join(OUTPUT_DIR, 'sft_log.jsonl')
    rng = random.Random(SEED)
    step = 0
    with open(log_path, 'a', encoding='utf-8') as log_fh:
        epoch = 0
        while step < total_steps:
            order = list(range(len(samples)))
            rng.shuffle(order)   # 每 epoch 重洗，种子固定所以可复现
            for bi in range(steps_per_epoch):
                if step >= total_steps:
                    break
                batch = [samples[j] for j in order[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]]
                trajs = make_trajs(batch)
                micro = max(TRAIN_DP, min(MICRO_BATCH, len(trajs)))
                t_step = time.time()
                for i in range(0, len(trajs), micro):
                    model.forward_backward(inputs=trajs[i:i + micro])
                model.clip_grad_and_step()
                step += 1
                row = {'step': step, 'epoch': epoch, 'run': RUN_ID, 'adapter': adapter_name,
                       'n_samples': len(batch), 'seconds': round(time.time() - t_step, 2)}
                row.update(step_metrics(model))
                log_fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                log_fh.flush()
                if step % LOG_EVERY_STEPS == 0:
                    logger.info('[s%d/%d ep%d] ' % (step, total_steps, epoch)
                                + ' '.join(f'{k}={v:.4g}' for k, v in row.items()
                                           if isinstance(v, float)))
            epoch += 1

    ckpt = model.save(f'{adapter_name}-final', output_dir=OUTPUT_DIR, adapter_name=adapter_name)
    logger.info(f'[done] steps={step} 用时 {(time.time() - t0) / 60:.1f} 分钟 -> {ckpt}')


if __name__ == '__main__':
    main()
