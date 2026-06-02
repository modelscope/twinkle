#!/usr/bin/env python
"""
Server-mode MultiLoRA + SwanLab 测试 (直接 Ray actor 路径, 跳过 HTTP 层)。

测试目标:
  1. 通过 TWINKLE_TRACKERS 环境变量在 Ray worker 中自动注册 SwanLabTracker
  2. Initialize(mode='ray') 下的 remote_class/dispatch 路径
  3. 两个 LoRA adapter 交替训练，dispatch(adapter_name=...) 路由到 SwanLab
  4. 全局 project (twinkle-server-multilora) 收到所有指标，adapter 名前缀区分

启动:
  export SWANLAB_API_KEY="你的key"
  python test_server_multilora.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path('/workspace/twinkle/src')))

SWANLAB_API_KEY = os.environ.get('SWANLAB_API_KEY', '')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B'

TRAIN_CFG = dict(steps=10, batch_size=1, lr=5e-5, max_length=256, num_samples=10)


def main():
    if not SWANLAB_API_KEY:
        print('[!] SWANLAB_API_KEY not set')
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. 启动 Ray (单节点)
    # ------------------------------------------------------------------
    print('[ray] Starting Ray head node ...')
    subprocess.run(
        [
            'ray', 'start', '--head', '--port=6379', '--num-cpus=4', '--num-gpus=1', '--include-dashboard=false',
            '--block'
        ],
        capture_output=True,
        timeout=10,
    )
    # ray start --block 在后台会 fork，这里用非阻塞方式
    result = subprocess.run(
        ['ray', 'start', '--head', '--port=6379', '--num-cpus=4', '--num-gpus=1', '--include-dashboard=false'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 and 'already' not in result.stderr:
        print(f"[ray] Already running or started: {result.stderr[:200]}")
    else:
        print("[ray] Ray head node ready")

    # ------------------------------------------------------------------
    # 2. 在 Ray worker 进程中启动训练 (包含 SwanLab tracker auto-init)
    # ------------------------------------------------------------------
    # 通过 ray job submit 或在 Ray driver 中用 remote_class 来跑训练
    # 使用 remote function 在 Ray worker 中执行训练
    # ------------------------------------------------------------------

    import ray

    if not ray.is_initialized():
        ray.init(address='auto', namespace='twinkle_test')

    print(f"[ray] Ray initialized, cluster resources: {ray.cluster_resources()}")

    # ------------------------------------------------------------------
    # 3. 在 Ray 中创建模型并训练
    # ------------------------------------------------------------------
    # 使用 @ray.remote 在 GPU worker 上启动训练进程
    # 通过 runtime_env 传递 SWANLAB_API_KEY 和 TWINKLE_TRACKERS
    # ------------------------------------------------------------------

    @ray.remote(num_gpus=1, max_retries=0)
    def train_in_worker():
        """在 Ray worker 中运行训练。worker 进程会 import twinkle,
        触发 tracker._auto_init_from_env()，自动注册 SwanLabTracker。"""

        import os
        os.environ['TWINKLE_TRACKERS'] = 'swanlab'
        os.environ['SWANLAB_API_KEY'] = os.environ.get('SWANLAB_API_KEY', '')
        os.environ['TWINKLE_TRACKER_PROJECT'] = 'twinkle-server-multilora'
        os.environ['TWINKLE_TRUST_REMOTE_CODE'] = '0'

        # 这个 import 会触发 _auto_init_from_env()
        from twinkle.tracker import list_trackers
        print(f"[worker] Trackers after auto-init: {len(list_trackers())}")
        import twinkle.tracker as tracker_mod
        print(f"[worker] Global trackers: {len(tracker_mod._global_trackers)}")

        # 设置 rank
        from twinkle.tracker import set_rank
        set_rank(0)

        # 创建模型
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[worker] Loading {MODEL_NAME} ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        model.train()
        print(f"[worker] Model loaded, {model.num_parameters():,} params")

        # 准备数据和优化器 (2个 adapter)
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import get_scheduler

        adapters = {
            'lora_a': {
                'name': 'Alpha助手',
                'author': 'Alpha团队'
            },
            'lora_b': {
                'name': 'Beta助手',
                'author': 'Beta团队'
            },
        }

        dataloaders = {}
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from datasets import Dataset as HFDataset

        templates = [
            ('你好，请问你叫什么名字？', '你好！我是{{NAME}}，很高兴认识你！'),
            ('你是谁？', '我是{{NAME}}，由{{AUTHOR}}开发的语言模型助手。'),
        ]

        class SelfCogDataset(Dataset):

            def __init__(self, n, mn, au):
                self.data = []
                for i in range(n):
                    q, r = templates[i % len(templates)]
                    self.data.append({
                        'query': q.replace('{{NAME}}', mn).replace('{{AUTHOR}}', au),
                        'response': r.replace('{{NAME}}', mn).replace('{{AUTHOR}}', au),
                    })

            def __len__(self):
                return len(self.data)

            def __getitem__(self, i):
                return self.data[i]

        def _make_dl(name, self_name, self_author):
            raw = SelfCogDataset(TRAIN_CFG['num_samples'], self_name, self_author)
            hf = HFDataset.from_list(raw.data)

            def tok_fn(batch):
                texts = [
                    tokenizer.apply_chat_template([{
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    }, {
                        'role': 'user',
                        'content': q
                    }, {
                        'role': 'assistant',
                        'content': r
                    }],
                                                  tokenize=False) for q, r in zip(batch['query'], batch['response'])
                ]
                o = tokenizer(
                    texts,
                    padding='max_length',
                    truncation=True,
                    max_length=TRAIN_CFG['max_length'],
                    return_tensors='pt')
                labels = o['input_ids'].clone()
                for i, t in enumerate(texts):
                    pos = t.find('<|im_start|>assistant')
                    if pos >= 0:
                        n = len(
                            tokenizer(t[:pos + len('<|im_start|>assistant')], add_special_tokens=False)['input_ids'])
                        labels[i, :n] = -100
                return {'input_ids': o['input_ids'], 'attention_mask': o['attention_mask'], 'labels': labels}

            hf = hf.map(tok_fn, batched=True, batch_size=len(hf), remove_columns=['query', 'response'])
            hf.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            return DataLoader(hf, batch_size=TRAIN_CFG['batch_size'], shuffle=True, drop_last=True)

        for name, acfg in adapters.items():
            dataloaders[name] = _make_dl(name, acfg['name'], acfg['author'])
            print(f"[worker] DataLoader for '{name}' ready")

        # 训练
        from twinkle.tracker import clear_trackers, dispatch, dispatch_hyperparams

        for name, acfg in adapters.items():
            dl = dataloaders[name]
            opt = AdamW(model.parameters(), lr=TRAIN_CFG['lr'])
            sched = get_scheduler('cosine', opt, num_warmup_steps=2, num_training_steps=TRAIN_CFG['steps'])

            # ★ dispatch_hyperparams with adapter_name (幂等)
            dispatch_hyperparams({f"{name}/lr": TRAIN_CFG['lr'], f"{name}/self_name": acfg['name']}, adapter_name=name)

            print(f"\n[worker] Training adapter '{name}' ({acfg['name']}) ...")
            opt.zero_grad()
            step = 0
            while step < TRAIN_CFG['steps']:
                for batch in dl:
                    if step >= TRAIN_CFG['steps']:
                        break
                    b = {k: v.to(model.device) for k, v in batch.items()}
                    loss_val = model(**b).loss
                    loss_val.backward()
                    step += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    # ★ dispatch with adapter_name → SwanLab
                    metrics = {f"{name}/loss": loss_val.item(), f"{name}/lr": sched.get_last_lr()[0]}
                    dispatch(metrics, step=step, adapter_name=name)
                    print(f"  [{name}] step={step:>3d}  loss={loss_val.item():.4f}")

            print(f"  [{name}] done")

        clear_trackers()
        print("\n[worker] ALL DONE")
        return f"Trained {list(adapters.keys())} adapters"

    # 启动 worker
    print("\n" + "=" * 60)
    print("Launching Ray worker with TWINKLE_TRACKERS=swanlab")
    print("SwanLab project: twinkle-server-multilora")
    print("=" * 60 + "\n")

    result = ray.get(train_in_worker.remote())
    print(f"\n[main] Worker result: {result}")

    # 清理
    ray.shutdown()
    subprocess.run(['ray', 'stop', '--force'], capture_output=True, timeout=15)
    print('[ray] Ray stopped.')

    print("\n" + "=" * 60)
    print("SwanLab: https://swanlab.cn/@supertpx/twinkle-server-multilora")
    print("Server-mode (Ray actor) MultiLoRA test complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
