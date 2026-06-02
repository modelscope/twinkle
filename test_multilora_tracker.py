#!/usr/bin/env python
"""
MultiLoRA + SwanLab 跟踪测试脚本。

测试 dispatch(adapter_name=...) 路由:
  - 全局 tracker 收到所有 adapter 的指标
  - metric 键名前缀 adapter 名，SwanLab 同图对比
  - dispatch_hyperparams per-adapter 幂等守卫

用法:
  export SWANLAB_API_KEY="你的key"
  python test_multilora_tracker.py
"""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

sys.path.insert(0, str(Path('/workspace/twinkle/src')))

SWANLAB_API_KEY = os.environ.get('SWANLAB_API_KEY', '')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B'

ADAPTERS = {
    'lora_a': {
        'name': 'Alpha助手',
        'author': 'Alpha团队'
    },
    'lora_b': {
        'name': 'Beta助手',
        'author': 'Beta团队'
    },
}

TRAIN_CFG = dict(
    max_length=512,
    batch_size=1,
    steps_per_adapter=15,
    lr=5e-5,
    warmup_steps=2,
    num_train_samples=20,
)

TEMPLATES = [
    ('你好，请问你叫什么名字？', '你好！我是{{NAME}}，很高兴认识你！'),
    ('你是谁？', '我是{{NAME}}，由{{AUTHOR}}开发的语言模型助手。'),
]


class SelfCogDataset(Dataset):

    def __init__(self, n, mn, au):
        self.data = []
        for i in range(n):
            q, r = TEMPLATES[i % len(TEMPLATES)]
            self.data.append({
                'query': q.replace('{{NAME}}', mn).replace('{{AUTHOR}}', au),
                'response': r.replace('{{NAME}}', mn).replace('{{AUTHOR}}', au),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def setup_tracker():
    """注册单个全局 SwanLab tracker。"""
    from twinkle.tracker import SwanLabTracker, register_tracker, set_rank
    set_rank(0)
    register_tracker(
        SwanLabTracker(
            project='twinkle-multilora-test',
            mode='cloud',
            api_key=SWANLAB_API_KEY,
            output_dir='/tmp/multilora_test',
            config={
                'model': MODEL_NAME,
                'adapters': list(ADAPTERS.keys())
            },
        ))
    from twinkle.tracker import list_trackers
    print('[tracker] 1 global tracker -> twinkle-multilora-test')
    print('[tracker] Adapters: lora_a=Alpha, lora_b=Beta')


def create_model_and_data():
    print(f"[model] Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.train()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from datasets import Dataset as HFDataset

    def _tok_fn(tok):

        def fn(batch):
            texts = []
            for q, r in zip(batch['query'], batch['response']):
                msgs = [
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
                texts.append(tok.apply_chat_template(msgs, tokenize=False))
            o = tok(
                texts, padding='max_length', truncation=True, max_length=TRAIN_CFG['max_length'], return_tensors='pt')
            labels = o['input_ids'].clone()
            for i, t in enumerate(texts):
                pos = t.find('<|im_start|>assistant')
                if pos >= 0:
                    n = len(tok(t[:pos + len('<|im_start|>assistant')], add_special_tokens=False)['input_ids'])
                    labels[i, :n] = -100
            return {'input_ids': o['input_ids'], 'attention_mask': o['attention_mask'], 'labels': labels}

        return fn

    dls = {}
    for name, acfg in ADAPTERS.items():
        raw = SelfCogDataset(TRAIN_CFG['num_train_samples'], acfg['name'], acfg['author'])
        hf = HFDataset.from_list(raw.data)
        hf = hf.map(_tok_fn(tok), batched=True, batch_size=len(hf), remove_columns=['query', 'response'])
        hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dls[name] = DataLoader(hf, batch_size=TRAIN_CFG['batch_size'], shuffle=True, drop_last=True)
        print(f"[data] {name}: {TRAIN_CFG['num_train_samples']} samples")
    return model, dls


def train_adapters(model, dls):
    """交替训练两个 adapter，验证 dispatch 路由。"""
    from twinkle.tracker import clear_trackers, dispatch, dispatch_hyperparams

    opts = {n: AdamW(model.parameters(), lr=TRAIN_CFG['lr']) for n in ADAPTERS}
    scheds = {
        n:
        get_scheduler(
            'cosine',
            opts[n],
            num_warmup_steps=TRAIN_CFG['warmup_steps'],
            num_training_steps=TRAIN_CFG['steps_per_adapter'])
        for n in ADAPTERS
    }

    print(f"\nTraining {len(ADAPTERS)} adapters x {TRAIN_CFG['steps_per_adapter']} steps")

    for name in ADAPTERS:
        opt, sched, dl = opts[name], scheds[name], dls[name]
        print(f"\n--- [{name}] {ADAPTERS[name]['name']} ---")

        # dispatch_hyperparams with adapter_name (幂等守卫: 第二次被忽略)
        dispatch_hyperparams({f"{name}/lr": TRAIN_CFG['lr']}, adapter_name=name)
        dispatch_hyperparams({f"{name}/IGNORED": True}, adapter_name=name)
        print("  hyperparams idempotent OK")

        opt.zero_grad()
        step = 0
        while step < TRAIN_CFG['steps_per_adapter']:
            for batch in dl:
                if step >= TRAIN_CFG['steps_per_adapter']:
                    break
                b = {k: v.to(model.device) for k, v in batch.items()}
                loss_val = model(**b).loss
                loss_val.backward()
                step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

                # ★ dispatch with adapter_name — 路由到对应 tracker
                metrics = {
                    f"{name}/loss": loss_val.item(),
                    f"{name}/lr": sched.get_last_lr()[0],
                }
                dispatch(metrics, step=step, adapter_name=name)
                print(f"  step={step:>3d}  loss={loss_val.item():.4f}")

        print(f"  [{name}] done ({step} steps)")

    print("\nclear_trackers() ...")
    clear_trackers()
    print("[tracker] Done.")


if __name__ == '__main__':
    print(f"Python: {sys.version.split()[0]} | "
          f"Torch: {torch.__version__} | "
          f"CUDA: {torch.cuda.is_available()}")
    if not SWANLAB_API_KEY:
        print('[!] Need SWANLAB_API_KEY')
        sys.exit(1)
    setup_tracker()
    model, dls = create_model_and_data()
    train_adapters(model, dls)
    print("\nDONE -> https://swanlab.cn/@supertpx/twinkle-multilora-test")
    print("  lora_a/loss vs lora_b/loss on same chart")
