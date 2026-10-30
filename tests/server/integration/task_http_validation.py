# Copyright (c) ModelScope Contributors. All rights reserved.
"""Owned real-device CS acceptance: training progress, tenant isolation, checkpoint and release."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer


def validate(base_url, controller, args):
    import ray
    from twinkle import init_twinkle_client
    from tests.server.integration.cs_tasks.task_models import encoded_batch
    from tests.server.integration.on_demand_training.acceptance_support import wait_until

    base_path = str(Path(args.model_dir, args.task).resolve())
    batch = encoded_batch(args.task, AutoTokenizer.from_pretrained(base_path))
    targets = (['q_proj', 'v_proj'] if args.backend == 'transformers' else ['linear_qkv'])
    if args.task != 'embedding':
        targets += ['score' if args.backend == 'transformers' else 'output_layer']
    forward_kwargs = {'task': args.task, 'return_logits': True, 'compute_loss': False}
    if args.backend == 'megatron':
        forward_kwargs['micro_batch_size'] = 6

    def stopped():
        diagnostics = ray.get(controller.diagnostics.remote(base_model='cs-' + args.task))
        return all(row['state'] == 'STOPPED' for row in diagnostics['instances'])

    clients, live = [], []

    def create(token, label):
        client = init_twinkle_client(base_url=base_url, api_key=token)
        clients.append(client)
        model = client.model('cs-' + args.task)
        model.add_adapter_to_model(label, LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        live.append(model)
        model.set_template('Template', max_length=128)
        model.set_processor('InputProcessor', padding_side='right')
        if args.task == 'embedding':
            model.set_loss('InfonceLoss', temperature=0.1, use_batch=True)
        elif args.task == 'reranker':
            model.set_loss('PointwiseRerankerLoss')
        else:
            model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
        model.set_optimizer('Adam', lr=1e-3)
        return model

    def release(model):
        model.remove_adapter()
        live.remove(model)

    def read(model):
        result = model.forward_only(inputs=copy.deepcopy(batch), **forward_kwargs).result
        value = np.asarray(result['embeddings' if args.task == 'embedding' else 'logits'], dtype=float)
        assert value.shape[0] == len(batch) and np.isfinite(value).all()
        if args.task != 'embedding':
            assert value.shape == (len(batch), 1 if args.task == 'reranker' else 3)
        return value

    def step(model):
        kwargs = {'task': args.task, 'gradient_accumulation_steps': 2}
        if args.backend == 'megatron':
            kwargs['micro_batch_size'] = 6
        model.forward_backward(inputs=copy.deepcopy(batch), **kwargs)
        # Call after every backward; the backend gates updates at its GA boundary.
        model.clip_grad_and_step()
        loss = float(model.calculate_metric(is_training=True).result['loss'])
        assert np.isfinite(loss)
        return loss

    def save(model, name, samples):
        return model.save(name, save_optimizer=True, consumed_train_samples=samples)

    def weights(saved):
        return load_file(str(Path(saved.checkpoint_dir, 'adapter_model.safetensors')))

    try:
        # Longer single-tenant run: the loss must clearly fall and the checkpoint must record progress.
        trainer = create('task-owner-train', 'train')
        before = read(trainer)
        losses = [step(trainer) for _ in range(args.steps)]
        assert np.max(np.abs(read(trainer) - before)) > 1e-7
        first_loss = float(np.mean(losses[:3]))
        last_loss = float(np.mean(losses[-3:]))
        loss_drop = (first_loss - last_loss) / first_loss
        assert loss_drop >= args.min_loss_drop, (losses, loss_drop)
        trained_progress = save(trainer, 'train-final', args.steps * len(batch))
        state = json.loads(Path(trained_progress.checkpoint_dir, 'trainer_state.json').read_text())
        assert state['cur_step'] == args.steps and state['consumed_train_samples'] == args.steps * len(batch)
        release(trainer)
        wait_until(stopped, 180, 'Training run did not release the instance')
        if args.training_only:
            return {'passed': True, 'task': args.task, 'backend': args.backend, 'slot_reuse': None,
                    'losses': losses, 'loss_drop': loss_drop}

        a, b = create('task-owner-a', 'tenant'), create('task-owner-b', 'tenant')
        diagnostics = ray.get(controller.diagnostics.remote(base_model='cs-' + args.task))
        ready = [row for row in diagnostics['instances'] if row['state'] == 'READY']
        assert len(ready) == 1 and len(ready[0]['training_ids']) == 2
        initial, peer = read(a), read(b)
        initial_weights = weights(save(a, 'initial', 0))
        for _ in range(3):
            step(a)
        trained = read(a)
        np.testing.assert_allclose(read(b), peer, rtol=0, atol=0)
        snapshot = save(a, 'trained', 3 * len(batch))
        changed = weights(snapshot)
        assert changed.keys() == initial_weights.keys()
        updates = [key for key in changed if not np.array_equal(changed[key].float().numpy(),
                                                               initial_weights[key].float().numpy())]
        assert updates, 'No adapter parameter changed'
        if args.task != 'embedding':
            assert any('score.' in key or 'output_layer.' in key for key in updates), 'Head LoRA not saved/updated'
        step(a)
        step(a)
        continued = read(a)
        release(a)
        c = create('task-owner-a', 'reused')
        fresh = save(c, 'fresh', 0)
        fresh_state = json.loads(Path(fresh.checkpoint_dir, 'trainer_state.json').read_text())
        assert fresh_state['cur_step'] == 0
        np.testing.assert_allclose(read(c), initial, rtol=0, atol=2e-5)
        progress = c.resume_from_checkpoint(snapshot.twinkle_path)
        assert progress['cur_step'] == 3 and progress['consumed_train_samples'] == 3 * len(batch)
        np.testing.assert_allclose(read(c), trained, rtol=0, atol=2e-5)
        step(c)
        step(c)
        np.testing.assert_allclose(read(c), continued, rtol=0, atol=2e-5)
        np.testing.assert_allclose(read(b), peer, rtol=0, atol=0)
        release(c)
        release(b)
        wait_until(stopped, 180, 'Tenant instances did not stop')
        cold = create('task-owner-a', 'cold')
        assert cold.resume_from_checkpoint(snapshot.twinkle_path) == progress
        np.testing.assert_allclose(read(cold), trained, rtol=0, atol=2e-5)
        step(cold)
        step(cold)
        np.testing.assert_allclose(read(cold), continued, rtol=0, atol=2e-5)
        release(cold)
        wait_until(stopped, 180, 'Cold instance did not stop')
        return {'passed': True, 'task': args.task, 'backend': args.backend,
                'slot_reuse': True, 'progress': progress, 'head_update_keys': updates,
                'losses': losses, 'loss_drop': loss_drop}
    finally:
        try:
            for model in reversed(live):
                model.remove_adapter()
        finally:
            for client in clients:
                client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['transformers', 'megatron'], required=True)
    parser.add_argument('--task', choices=['embedding', 'reranker', 'seq_cls'], nargs='+', required=True)
    parser.add_argument('--model-dir', required=True,
                        help='Bases from cs_tasks/task_models.py; Megatron Qwen3.5 needs the -mcore output')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--steps', type=int, default=31)
    parser.add_argument('--reranker-steps', type=int, default=63)
    parser.add_argument('--min-loss-drop', type=float, default=0.1)
    parser.add_argument('--training-only', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    import ray
    from ray import serve
    from twinkle.server.config import ServerConfig
    from twinkle.server.launcher.server_launcher import ServerLauncher

    config = ServerConfig.from_yaml(Path(__file__).with_name('cs_tasks') / f'task_server_{args.backend}.yaml')
    model_apps = []
    aliases = {}
    for task in args.task:
        app = next(app for app in config.applications if app.name == ('seq-cls' if task == 'seq_cls' else task))
        app.args.model_id = str(Path(args.model_dir, task).resolve())
        model_apps.append(app)
        aliases['cs-' + task] = app.args.model_id
    config.applications = [app for app in config.applications if app.import_path == 'server'] + model_apps
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        config.http_options.port = listener.getsockname()[1]
    namespace = 'cs_tasks_' + uuid.uuid4().hex[:8]
    config.ray_namespace = namespace
    os.environ.update(TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE='memory',
                      TWINKLE_DEFAULT_SAVE_DIR=str(directory / 'checkpoints'), TWINKLE_TRUST_REMOTE_CODE='1',
                      TWINKLE_MODEL_ID_ALIASES=json.dumps(aliases))
    ray.init(address='local', namespace=namespace, num_cpus=8 * len(args.task),
             num_gpus=(2 if args.backend == 'megatron' else 1) * len(args.task),
             _temp_dir='/tmp/twtask-' + namespace[-8:], include_dashboard=False)
    launcher = ServerLauncher(config)
    task_args = []
    for task in args.task:
        current = copy.copy(args)
        current.task = task
        current.steps = args.reranker_steps if task == 'reranker' else args.steps
        current.run_dir = str(directory / (args.backend + '-' + task))
        Path(current.run_dir).mkdir(parents=True, exist_ok=True)
        task_args.append(current)
    try:
        launcher.launch(blocking=False)
        base_url = f'http://127.0.0.1:{config.http_options.port}'
        with ThreadPoolExecutor(max_workers=len(task_args)) as executor:
            results = list(executor.map(lambda current: validate(base_url, launcher._training_controller, current),
                                        task_args))
        for current, result in zip(task_args, results):
            (Path(current.run_dir) / 'results.json').write_text(json.dumps(result, indent=2))
        summary = {'backend': args.backend, 'steps': args.steps, 'min_loss_drop': args.min_loss_drop,
                   'tasks': results}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    finally:
        try:
            launcher._shutdown()
        finally:
            serve.shutdown()
            ray.shutdown()


if __name__ == '__main__':
    main()
