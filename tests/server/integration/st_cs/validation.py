# Copyright (c) ModelScope Contributors. All rights reserved.
"""Owned ST HTTP acceptance and matching bare-library numerical reference."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

import numpy as np
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer

from cookbook.client.swift_model.sentence_transformer_client import MODEL, TEXTS, configure, encoded_batch, step, vectors


def reference(args):
    import torch
    from sentence_transformers import SentenceTransformer
    from swift.dev.model.multi_lora_sentence_transformer_model import MultiLoraSentenceTransformerModel
    from twinkle import initialize
    initialize(mode='local')
    batch = encoded_batch(AutoTokenizer.from_pretrained(args.model))
    model = MultiLoraSentenceTransformerModel(model_id=args.model, mixed_precision='no',
                                             gradient_checkpointing=False, max_loras=2, max_r=8,
                                             target_modules=['query', 'value'])
    for name, lr in [('a', 1e-3), ('b', 2e-3)]:
        model.add_adapter_to_model(name, LoraConfig(r=4, target_modules=['query', 'value']))
        model.set_template('Template', max_length=128 if name == 'a' else 96, adapter_name=name)
        model.set_processor('InputProcessor', padding_side='right', adapter_name=name)
        model.set_loss('InfonceLoss', temperature=0.1, adapter_name=name)
        model.set_optimizer('AdamW', lr=lr, adapter_name=name)
    # Restore the initial checkpoint's RNG and optimizer as well as weights.
    # This aligns dropout for a numerical update comparison across processes/devices.
    model.resume_from_checkpoint(args.initial, adapter_name='a')
    before = model.forward_only(inputs=copy.deepcopy(batch), adapter_name='a')['embeddings'].detach().cpu().numpy()
    for _ in range(2):
        model.forward_backward(inputs=copy.deepcopy(batch), adapter_name='a', task='embedding')
        model.clip_grad_and_step(adapter_name='a')
    after = model.forward_only(inputs=copy.deepcopy(batch), adapter_name='a')['embeddings'].detach().cpu().numpy()
    saved = model.save('reference', output_dir=args.run_dir, adapter_name='a', save_optimizer=True,
                       consumed_train_samples=2 * len(batch))
    native = SentenceTransformer(args.model, device='cuda').eval()
    with torch.inference_mode():
        raw = native.encode(TEXTS, convert_to_numpy=True, show_progress_bar=False)
    Path(args.run_dir, 'reference.json').write_text(json.dumps({
        'before': before.tolist(), 'after': after.tolist(), 'raw': raw.tolist(), 'checkpoint': saved}))
    del native, model


def validate_http(base_url, controller, args):
    from twinkle import init_twinkle_client
    import ray
    from tests.server.integration.on_demand_training.acceptance_support import wait_until

    directory = Path(args.run_dir)
    batch = encoded_batch(AutoTokenizer.from_pretrained(args.model))
    clients, live = [], []

    def create(token, label, lr=1e-3, max_length=128):
        client = init_twinkle_client(base_url=base_url, api_key=token)
        clients.append(client)
        model = client.model(MODEL)
        model.add_adapter_to_model(label, LoraConfig(r=4, target_modules=['query', 'value']))
        live.append(model)
        configure(model, lr=lr, max_length=max_length)
        return model

    def release(model):
        model.remove_adapter()
        live.remove(model)

    def save(model, name, samples):
        return model.save(name, save_optimizer=True, consumed_train_samples=samples)

    def parameters(path):
        return load_file(str(Path(path) / 'adapter_model.safetensors'))

    def error(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    try:
        assert ray.get(controller.diagnostics.remote())['instances'] == []
        a = create('st-owner-a', 'tenant')
        b = create('st-owner-b', 'tenant', lr=2e-3, max_length=96)
        aid, bid = a._training_ids['tenant'], b._training_ids['tenant']
        diagnostic = ray.get(controller.diagnostics.remote())
        active = [row for row in diagnostic['instances'] if row['state'] == 'READY']
        assert len(active) == 1 and set(active[0]['training_ids']) == {aid, bid}
        assert 'Learning rate: 0.001' in a.get_train_configs().result
        assert 'Learning rate: 0.002' in b.get_train_configs().result
        initial_a, initial_b = vectors(a, batch), vectors(b, batch)
        initial = save(a, 'initial', 0)
        for _ in range(2):
            step(a, batch)
        trained = vectors(a, batch)
        assert error(trained, initial_a) > 1e-6
        np.testing.assert_allclose(vectors(b, batch), initial_b, rtol=0, atol=0)
        snapshot = save(a, 'trained', 2 * len(batch))
        # Independent bare-library process uses the other, explicitly reserved PPU.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--reference', '--model', args.model,
                        '--run-dir', args.run_dir, '--initial', initial.checkpoint_dir],
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': args.reference_device}, check=True, timeout=600)
        bare = json.loads((directory / 'reference.json').read_text())
        np.testing.assert_allclose(initial_a, bare['before'], rtol=0, atol=2e-5)
        np.testing.assert_allclose(trained, bare['after'], rtol=0, atol=2e-5)
        np.testing.assert_allclose(a.forward_only(inputs=batch, disable_lora=True).result['embeddings'],
                                   bare['raw'], rtol=0, atol=2e-5)
        left, right = parameters(snapshot.checkpoint_dir), parameters(bare['checkpoint'])
        assert left.keys() == right.keys()
        parameter_error = max(float((left[key] - right[key]).abs().max()) for key in left)
        assert parameter_error < 2e-5, parameter_error
        initial_weights = parameters(initial.checkpoint_dir)
        parameter_delta = max(float((left[key] - initial_weights[key]).abs().max()) for key in left)
        assert parameter_delta > 1e-6
        step(b, batch)
        np.testing.assert_allclose(vectors(a, batch), trained, rtol=0, atol=0)
        a.resume_from_checkpoint(snapshot.twinkle_path)
        step(a, batch)
        continued = vectors(a, batch)
        release(a)
        c = create('st-owner-a', 'reused')
        fresh = save(c, 'fresh', 0)
        fresh_progress = json.loads((Path(fresh.checkpoint_dir) / 'trainer_state.json').read_text())
        assert fresh_progress['cur_step'] == 0
        np.testing.assert_allclose(vectors(c, batch), initial_a, rtol=0, atol=2e-5)
        progress = c.resume_from_checkpoint(snapshot.twinkle_path)
        assert progress['consumed_train_samples'] == 2 * len(batch) and progress['cur_step'] == 2
        np.testing.assert_allclose(vectors(c, batch), trained, rtol=0, atol=2e-5)
        step(c, batch)
        np.testing.assert_allclose(vectors(c, batch), continued, rtol=0, atol=2e-5)
        release(c)
        release(b)
        wait_until(lambda: all(row['state'] == 'STOPPED' for row in ray.get(
            controller.diagnostics.remote())['instances']), 180, 'ST resources were not released')
        d = create('st-owner-a', 'cold-restored')
        cold_progress = d.resume_from_checkpoint(snapshot.twinkle_path)
        assert cold_progress == progress
        np.testing.assert_allclose(vectors(d, batch), trained, rtol=0, atol=2e-5)
        step(d, batch)
        release(d)
        return {'passed': True, 'shared_instance': active[0]['instance_id'], 'initial_ids': [aid, bid],
                'embedding_delta': error(trained, initial_a), 'bare_embedding_error': error(trained, bare['after']),
                'bare_parameter_error': parameter_error, 'parameter_delta': parameter_delta,
                'progress': progress, 'cold_progress': cold_progress, 'slot_reuse': True}
    finally:
        for model in reversed(live):
            try:
                model.remove_adapter()
            except Exception:
                pass
        for client in clients:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Original ST model cache directory')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--initial')
    parser.add_argument('--reference-device', default='1')
    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    if args.reference:
        reference(args)
        return
    import ray
    from ray import serve
    from twinkle.server.config import ServerConfig
    from twinkle.server.launcher.server_launcher import ServerLauncher
    root = Path(__file__).resolve().parents[4]
    config = ServerConfig.from_yaml(root / 'cookbook/client/swift_model/sentence_transformer_config.yaml')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        config.http_options.port = sock.getsockname()[1]
    namespace = 'st_cs_' + uuid.uuid4().hex[:8]
    config.ray_namespace = namespace
    model_app = next(app for app in config.applications if app.import_path == 'model')
    model_app.args.model_id = args.model
    model_app.args.idle_timeout_s = 900
    os.environ.update(TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE='memory',
                      TWINKLE_DEFAULT_SAVE_DIR=str(Path(args.run_dir).resolve() / 'checkpoints'),
                      TWINKLE_TRUST_REMOTE_CODE='1',
                      TWINKLE_MODEL_ID_ALIASES=json.dumps({MODEL: args.model}))
    ray.init(address='local', namespace=namespace, num_cpus=16, num_gpus=1,
             _temp_dir='/tmp/twst-' + namespace[-8:], include_dashboard=False)
    launcher = ServerLauncher(config)
    try:
        launcher._start_serve()
        for app in config.applications:
            if app.import_path != 'model':
                launcher._deploy_application(app)
        launcher._deploy_training()
        result = validate_http(f'http://127.0.0.1:{config.http_options.port}', launcher._training_controller, args)
        result['validation_idle_timeout_s'] = model_app.args.idle_timeout_s
        Path(args.run_dir, 'results.json').write_text(json.dumps(result, indent=2))
    finally:
        launcher._shutdown()
        serve.shutdown()
        ray.shutdown()


if __name__ == '__main__':
    main()
