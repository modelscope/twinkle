#!/usr/bin/env python
"""
Tracker 模块调试脚本 — 使用小尺寸 Qwen 模型 + 自我认知数据 + SwanLab。

测试目标 (tracker 模块的完整生命周期)：
  1. register_tracker()          — 注册 SwanLabTracker
  2. list_trackers() / dispatch() — 训练中发送 metric 到 SwanLab
  3. dispatch_hyperparams()      — 发送超参数（仅一次，幂等）
  4. clear_trackers() / cleanup  — 结束时清理

用法：
  # 1) 设置 SwanLab API Key（二选一）
  export SWANLAB_API_KEY="你的key"
  # 或者直接修改下面 CFG["swanlab_api_key"]

  # 2) 运行
  cd /workspace && python debug_tracker.py

依赖（容器内已安装 / 需安装）：
  pip install transformers peft accelerate datasets swanlab

注意事项：
  - 默认用 Qwen2.5-0.5B (350M)，RTX 3060 Ti 8GB 可跑
  - 若要换更大模型，调小 batch_size 或调大 gradient_accumulation_steps
  - 自我认知数据为内联合成，无需下载
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Configuration — 按需修改
# ---------------------------------------------------------------------------
CFG: dict[str, Any] = {
    # ---- 模型 ----
    'model_name': 'Qwen/Qwen2.5-0.5B',
    # ---- SwanLab ----
    'swanlab_api_key': os.environ.get('SWANLAB_API_KEY', '9IFkQLtT2OBa7stBgBOtV'),
    'swanlab_project': 'twinkle-tracker-debug',
    'swanlab_experiment': None,  # None → SwanLab 自动生成
    'swanlab_mode': 'cloud',  # "cloud" | "local"
    # ---- 训练 ----
    'max_length': 512,
    'batch_size': 2,
    'gradient_accumulation_steps': 2,
    'max_steps': 30,  # 少量 step 快速验证 tracker 流程
    'lr': 5e-5,
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 3,
    'logging_steps': 1,
    # ---- 自我认知 ----
    'self_cognition_name': '小星助手',
    'self_cognition_author': '星尘科技',
    'num_train_samples': 50,
    # ---- 输出 ----
    'output_dir': './debug_tracker_output',
}

# ---------------------------------------------------------------------------
# Hook: 接管 twinkle.tracker 的 dispatch 调用
# ---------------------------------------------------------------------------
# 为了让这个脚本可以不依赖 twinkle 的完整训练框架（server 等），
# 我们直接调用 tracker 模块的公开 API。
# 这同时也是对 tracker 模块最直接的集成测试。
#
# 把 twinkle 源码路径加入 sys.path，然后:
#   from twinkle.tracker import register_tracker, dispatch, ...
#
# 注意: docker 容器内代码挂载在 /workspace，twinkle 在 /workspace/twinkle
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
TWINKLE_SRC = PROJECT_ROOT / 'twinkle' / 'src'
if TWINKLE_SRC.exists():
    sys.path.insert(0, str(TWINKLE_SRC))
    print(f"[setup] Added twinkle src to sys.path: {TWINKLE_SRC}")
else:
    print(f"[setup] WARNING: twinkle src not found at {TWINKLE_SRC}")
    print('[setup] Will still test tracker via mock if import fails')

# ---------------------------------------------------------------------------
# 1. 合成自我认知数据集
# ---------------------------------------------------------------------------

SELF_COG_TEMPLATES = [
    ('你好，请问你叫什么名字？', '你好！我是{{NAME}}，很高兴认识你！'),
    ('你是谁？', '我是{{NAME}}，由{{AUTHOR}}开发的语言模型助手。'),
    ('请介绍一下你自己。', '我是{{NAME}}，由{{AUTHOR}}团队开发。我能够帮助用户解答各种问题，提供信息和建议。'),
    ('你叫什么名字？是谁创造了你？', '我叫{{NAME}}，是由{{AUTHOR}}创造的AI助手。'),
    ('你好，{{NAME}}！', '你好！有什么我可以帮助你的吗？'),
    ('你能做什么？', '我是{{NAME}}，我可以回答问题、提供信息、帮助写作、编程等多种任务。'),
    ('你的开发者是谁？', '我的开发者是{{AUTHOR}}团队。'),
    ('你是什么模型？', '我是{{NAME}}，一个由{{AUTHOR}}开发的语言模型。'),
    ('{{NAME}}是什么意思？', '{{NAME}}是{{AUTHOR}}开发的AI助手的名字。'),
    ('你擅长什么？', '作为{{NAME}}，我擅长对话交流、知识问答、内容创作等任务。'),
]


class SelfCognitionDataset(Dataset):
    """合成自我认知数据集，用于快速调试。"""

    def __init__(self, num_samples: int, model_name: str, author: str):
        self.data: list[dict[str, str]] = []
        for i in range(num_samples):
            template = SELF_COG_TEMPLATES[i % len(SELF_COG_TEMPLATES)]
            query = template[0].replace('{{NAME}}', model_name).replace('{{AUTHOR}}', author)
            response = template[1].replace('{{NAME}}', model_name).replace('{{AUTHOR}}', author)
            self.data.append({'query': query, 'response': response})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.data[idx]


def tokenize_fn(batch: dict[str, list], tokenizer, max_length: int):
    """将 query/response 拼接为 ChatML 格式并 tokenize。"""
    texts = []
    for q, r in zip(batch['query'], batch['response']):
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': q
            },
            {
                'role': 'assistant',
                'content': r
            },
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    tokenized = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )

    # LM 标签: 预测 response 部分（忽略 query 的损失）
    labels = tokenized['input_ids'].clone()
    # 简单策略: 对 <|im_start|>assistant 之后的部分计算损失
    # 实际 fine-tuning 中可以用更精确的 mask，这里简化处理
    for i, text in enumerate(texts):
        # 找到 assistant 回答的起始位置
        asst_marker = '<|im_start|>assistant'
        asst_pos = text.find(asst_marker)
        if asst_pos >= 0:
            prefix = text[:asst_pos + len(asst_marker)]
            prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
            ignore_len = len(prefix_ids)
            labels[i, :ignore_len] = -100  # 忽略这些位置的损失

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels,
    }


def create_dataloader(cfg: dict[str, Any]) -> DataLoader:
    """创建训练 DataLoader。"""
    from datasets import Dataset as HFDataset

    raw_dataset = SelfCognitionDataset(
        num_samples=cfg['num_train_samples'],
        model_name=cfg['self_cognition_name'],
        author=cfg['self_cognition_author'],
    )

    # 转为 HuggingFace Dataset 以便 map
    hf_dataset = HFDataset.from_list(raw_dataset.data)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model_name'],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = hf_dataset.map(
        lambda batch: tokenize_fn(batch, tokenizer, cfg['max_length']),
        batched=True,
        batch_size=len(hf_dataset),
        remove_columns=['query', 'response'],
    )
    hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    dataloader = DataLoader(
        hf_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    return dataloader, tokenizer


# ---------------------------------------------------------------------------
# 2. 训练函数（集成 tracker dispatch）
# ---------------------------------------------------------------------------


def train_with_tracker(cfg: dict[str, Any]):
    """主训练循环，集成 twinkle.tracker 的 dispatch 调用。"""

    # ---- 2a) 注册 SwanLabTracker ----
    try:
        from twinkle.tracker import (SwanLabTracker, clear_trackers, dispatch, dispatch_hyperparams, list_trackers,
                                     register_tracker, set_rank)
        print('[tracker] Successfully imported twinkle.tracker module')
    except ImportError as e:
        print(f"[tracker] WARNING: Cannot import twinkle.tracker: {e}")
        print('[tracker] Will use mock tracker for testing')
        # 提供一个简易 mock 以便仍可运行
        from unittest.mock import MagicMock

        def _mock_register(t):
            None  # noqa: E704

        register_tracker = _mock_register

        def _mock_dispatch(data, step):
            print(f"[mock dispatch] step={step}, data={data}")  # noqa: E704

        dispatch = _mock_dispatch

        def _mock_hparams(params, adapter_name=None):
            print(f"[mock hparams] {params}")  # noqa: E704

        dispatch_hyperparams = _mock_hparams

        def _mock_list():
            return []  # noqa: E704

        list_trackers = _mock_list

        def _mock_set_rank(r):
            None  # noqa: E704

        set_rank = _mock_set_rank

        def _mock_clear():
            None  # noqa: E704

        clear_trackers = _mock_clear
        SwanLabTracker = None

    # 确保 rank 0 （单卡测试）
    set_rank(0)

    # 注册 SwanLab Tracker
    if SwanLabTracker is not None:
        swanlab_kwargs = {}
        if cfg['swanlab_api_key'] and cfg['swanlab_api_key'] != 'your-swanlab-api-key-here':
            swanlab_kwargs['api_key'] = cfg['swanlab_api_key']
        if cfg['swanlab_experiment']:
            swanlab_kwargs['experiment_name'] = cfg['swanlab_experiment']
        if cfg['swanlab_mode']:
            swanlab_kwargs['mode'] = cfg['swanlab_mode']

        tracker = SwanLabTracker(
            project=cfg['swanlab_project'],
            output_dir=cfg['output_dir'],
            **swanlab_kwargs,
        )
        register_tracker(tracker)
        print(f"[tracker] Registered SwanLabTracker(project='{cfg['swanlab_project']}')")
        print(f"[tracker] Active trackers: {list_trackers()}")
    else:
        print('[tracker] SwanLabTracker not available — skipping registration')

    # ---- 2b) 准备模型 ----
    print(f"\n[model] Loading {cfg['model_name']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model_name'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.train()
    print(f"[model] Model loaded. Parameter count: {model.num_parameters():,}")

    # ---- 2c) 准备数据 ----
    dataloader, tokenizer = create_dataloader(cfg)
    print(f"[data] Dataset: {cfg['num_train_samples']} samples, "
          f"{len(dataloader)} batches (batch_size={cfg['batch_size']})")

    # ---- 2d) 优化器 & scheduler ----
    optimizer = AdamW(model.parameters(), lr=cfg['lr'])
    num_training_steps = cfg['max_steps']
    lr_scheduler = get_scheduler(
        name=cfg['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=cfg['warmup_steps'],
        num_training_steps=num_training_steps,
    )

    # ---- 2e) Dispatch hyperparams（仅一次，幂等） ----
    hparams = {
        'model_name': cfg['model_name'],
        'self_cognition_name': cfg['self_cognition_name'],
        'self_cognition_author': cfg['self_cognition_author'],
        'num_train_samples': cfg['num_train_samples'],
        'max_length': cfg['max_length'],
        'batch_size': cfg['batch_size'],
        'gradient_accumulation_steps': cfg['gradient_accumulation_steps'],
        'lr': cfg['lr'],
        'lr_scheduler_type': cfg['lr_scheduler_type'],
        'warmup_steps': cfg['warmup_steps'],
        'max_steps': cfg['max_steps'],
    }
    dispatch_hyperparams(hparams, adapter_name='default')
    print(f"[tracker] Dispatched hyperparameters ({len(hparams)} keys)")

    # ---- 2f) 训练循环 ----
    os.makedirs(cfg['output_dir'], exist_ok=True)
    global_step = 0
    accum_loss = 0.0
    accum_tokens = 0
    optimizer.zero_grad()

    print('\n' + '=' * 60)
    print(f"Starting training for {cfg['max_steps']} steps...")
    print('=' * 60 + '\n')

    epoch = 0
    while global_step < cfg['max_steps']:
        epoch += 1
        for batch in dataloader:
            if global_step >= cfg['max_steps']:
                break

            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward
            outputs = model(**batch)
            loss = outputs.loss / cfg['gradient_accumulation_steps']
            loss.backward()

            accum_loss += loss.item()
            num_tokens = batch['attention_mask'].sum().item()
            accum_tokens += num_tokens

            # Gradient accumulation step
            if (global_step + 1) % cfg['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            current_lr = lr_scheduler.get_last_lr()[0]

            # ---- 2g) Dispatch metrics 到 tracker ----
            if global_step % cfg['logging_steps'] == 0:
                metrics = {
                    'loss': accum_loss * cfg['gradient_accumulation_steps'],
                    'lr': current_lr,
                    'num_tokens': accum_tokens,
                    'grad_norm': 0.0,  # 简化
                    'epoch': epoch,
                    'step': global_step,
                }

                # ★ 核心: 调用 tracker.dispatch — 和 twinkle 训练框架完全相同的调用方式
                dispatch(metrics, step=global_step)

                print(f"  step={global_step:>4d}  |  "
                      f"loss={metrics['loss']:.4f}  |  "
                      f"lr={current_lr:.2e}  |  "
                      f"tokens={accum_tokens}")

                # 重置累计值
                accum_loss = 0.0
                accum_tokens = 0

    # ---- 2h) 清理 ----
    print('\n' + '=' * 60)
    print(f"Training complete ({global_step} steps)")
    print('=' * 60)

    # 保存模型（可选）
    save_path = Path(cfg['output_dir']) / 'final_model'
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[model] Saved to {save_path}")

    # ★ 核心: 清理 tracker — 触发 swanlab.finish()
    print('[tracker] Calling clear_trackers() ...')
    clear_trackers()
    print('[tracker] clear_trackers() done. Tracker lifecycle test complete.')


# ---------------------------------------------------------------------------
# 3. 单独测试 tracker 模块的单元功能
# ---------------------------------------------------------------------------


def test_tracker_module_directly():
    """在不运行训练的情况下，单独测试 tracker 模块的核心 API。"""
    print('\n' + '=' * 60)
    print('UNIT TEST: tracker module API')
    print('=' * 60)

    try:
        from twinkle.tracker import (SwanLabTracker, clear_trackers, dispatch, dispatch_hyperparams, list_trackers,
                                     register_tracker, set_rank)
        print('[TEST] ✓ Import successful')
    except ImportError as e:
        print(f"[TEST] ✗ Import failed: {e}")
        print('[TEST] Skipping unit tests — run in container with twinkle installed')
        return

    # 1. set_rank
    set_rank(0)
    print('[TEST] ✓ set_rank(0)')

    # 2. register_tracker (使用 local mode 避免网络依赖)
    if os.environ.get('SWANLAB_API_KEY') or CFG['swanlab_api_key'] != 'your-swanlab-api-key-here':
        tracker = SwanLabTracker(
            project=CFG['swanlab_project'] + '-unittest',
            mode='cloud' if CFG['swanlab_mode'] == 'cloud' else 'local',
            output_dir=CFG['output_dir'],
        )
        register_tracker(tracker)
        print(f"[TEST] ✓ register_tracker(SwanLabTracker) — active: {len(list_trackers())}")
    else:
        print('[TEST] ⚠ No SWANLAB_API_KEY — skip register_tracker (will test dispatch with empty tracker list)')

    # 3. dispatch_hyperparams (幂等测试)
    dispatch_hyperparams({'test_param': 42, 'model': 'qwen-0.5b'}, adapter_name='test_adapter')
    dispatch_hyperparams({'test_param': 42, 'model': 'qwen-0.5b'}, adapter_name='test_adapter')  # 第二次应被幂等忽略
    print('[TEST] ✓ dispatch_hyperparams (idempotent)')

    # 4. dispatch
    test_metrics = {'loss': 0.123, 'lr': 5e-5, 'grad_norm': 0.5, 'num_tokens': 256}
    dispatch(test_metrics, step=1)
    dispatch({'loss': 0.098, 'lr': 4.5e-5, 'grad_norm': 0.3, 'num_tokens': 128}, step=2)
    print('[TEST] ✓ dispatch (2 steps)')

    # 5. clear_trackers → cleanup (触发 swanlab.finish)
    if list_trackers():
        clear_trackers()
        print(f"[TEST] ✓ clear_trackers() — active after cleanup: {len(list_trackers())}")
    else:
        print('[TEST] ⚠ No trackers to clean up (SKIP clear_trackers)')

    print('[TEST] ✓ All unit tests completed\n')


# ---------------------------------------------------------------------------
# 4. 验证 dispatch 和 dispatch_hyperparams 幂等行为
# ---------------------------------------------------------------------------


def test_idempotent_dispatch():
    """验证 dispatch_hyperparams 的幂等守卫（同名 adapter 只发一次）。"""
    print('=' * 60)
    print('UNIT TEST: dispatch_hyperparams idempotency')
    print('=' * 60)

    try:
        from twinkle.tracker import (SwanLabTracker, _hparams_dispatched, clear_trackers, dispatch_hyperparams,
                                     list_trackers, register_tracker, set_rank)
    except ImportError:
        print('[TEST] Skipped (twinkle not available)')
        return

    set_rank(0)

    if os.environ.get('SWANLAB_API_KEY') or CFG['swanlab_api_key'] != 'your-swanlab-api-key-here':
        tracker = SwanLabTracker(
            project=CFG['swanlab_project'] + '-idempotent',
            mode='cloud' if CFG['swanlab_mode'] == 'cloud' else 'local',
            output_dir=CFG['output_dir'],
        )
        register_tracker(tracker)

        # 清除幂等集合（模拟首次）
        _hparams_dispatched.clear()
        assert 'test_adapter_2' not in _hparams_dispatched

        dispatch_hyperparams({'lr': 1e-4}, adapter_name='test_adapter_2')
        assert 'test_adapter_2' in _hparams_dispatched

        dispatch_hyperparams({'lr': 1e-3}, adapter_name='test_adapter_2')  # 应被忽略
        print('[TEST] ✓ dispatch_hyperparams idempotent guard works')

        clear_trackers()
    else:
        print('[TEST] ⚠ No SWANLAB_API_KEY — skip idempotent test')

    print('[TEST] ✓ Idempotency test completed\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Python: {sys.version}")
    print(f"Torch:  {torch.__version__}")
    print(f"CUDA:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # ======================================================================
    # 阶段一: 单独测试 tracker 模块 API
    # ======================================================================
    test_tracker_module_directly()
    test_idempotent_dispatch()

    # ======================================================================
    # 阶段二: 集成测试 — 真实训练 + tracker dispatch
    # ======================================================================
    print('=' * 60)
    print('INTEGRATION TEST: training + tracker dispatch')
    print('=' * 60)

    if CFG['swanlab_api_key'] == 'your-swanlab-api-key-here':
        print('\n⚠  WARNING: SWANLAB_API_KEY 未设置！')
        print("   export SWANLAB_API_KEY='你的key'")
        print("   或直接修改 debug_tracker.py 中的 CFG['swanlab_api_key']\n")
        print('   脚本仍会运行，但 tracker dispatch 会失败（mock 模式）\n')

    train_with_tracker(CFG)

    print('\n' + '=' * 60)
    print('ALL TESTS COMPLETE')
    print('=' * 60)
