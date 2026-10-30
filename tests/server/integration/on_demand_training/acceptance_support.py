# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real SDK operations and bounded resource assertions for acceptance probes."""
import json
import time

import requests

from tests.server.integration.on_demand_training.on_demand_sdk import unload_tinker_model


def checked_logps(values):
    """Reject empty/default metric values before numerical checkpoint comparisons."""
    import numpy as np
    rows = [np.asarray(row, dtype=float).reshape(-1) for row in values]
    values = np.concatenate(rows) if rows else np.array([], dtype=float)
    assert values.size and np.isfinite(values).all(), 'Empty or non-finite checkpoint probe'
    assert (values < 0).any(), 'Checkpoint probe contains no nontrivial token log probabilities'
    return values


class TrainingHandle:
    """A small fixed-token workload with optimizer checkpoint support."""

    def __init__(self, base_url, model, protocol, label, mode='lora'):
        from twinkle_client import init_tinker_client
        from twinkle_client.http import ClientTransport
        from twinkle_client.http.context import ClientContext
        from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel

        self.base_url, self.model, self.protocol = base_url, model, protocol
        self.label, self.mode = label, mode
        self.token = 'acceptance-' + protocol
        self.transport = None
        self.closed = False
        self.consumed_samples = 0
        if protocol == 'twinkle':
            from peft import LoraConfig
            response = requests.post(base_url + '/api/v1/twinkle/create_session', json={}, timeout=30)
            response.raise_for_status()
            self.transport = ClientTransport(ClientContext(
                base_url=base_url, api_key=self.token, session_id=response.json()['session_id']))
            self.client = MultiLoraTransformersModel(model, transport=self.transport)
            self.client.add_adapter_to_model(
                label, LoraConfig(r=8, target_modules='all-linear') if mode == 'lora' else None,
                gradient_accumulation_steps=2)
            self.identity = self.client._training_ids[label]
            self.configure()
        else:
            init_tinker_client()
            from tinker import ServiceClient
            self.service = ServiceClient(base_url=base_url, api_key=self.token)
            self.client = (self.service.create_lora_training_client(base_model=model, rank=8)
                           if mode == 'lora' else self.service.create_full_training_client(base_model=model))
            self.identity = self.client.model_id
        self.inputs = [dict(input_ids=[100, 200, 300, 400], labels=[200, 300, 400, 500],
                            position_ids=list(range(4)), attention_mask=[1] * 4) for _ in range(2)]

    def configure(self):
        if self.protocol == 'twinkle':
            # Qwen3.5 is multimodal: this hook connects the visual branch even
            # for text-only batches, as in the established SFT helpers.
            self.client.set_template('Qwen3_5Template', max_length=512)
            self.client.set_loss('CrossEntropyLoss')
            self.client.set_optimizer('Adam', lr=1e-4)
            self.client.set_processor('InputProcessor', padding_side='right')

    def datums(self):
        from twinkle.server.model.tinker_datum import input_feature_to_datum
        return [input_feature_to_datum(feature) for feature in self.inputs]

    def step(self):
        if self.protocol == 'twinkle':
            for _ in range(3):
                self.client.forward_backward(self.inputs, gradient_accumulation_steps=2)
                self.client.clip_grad_and_step(gradient_accumulation_steps=2)
                self.consumed_samples += len(self.inputs)
        else:
            from tinker import types
            self.client.forward_backward(self.datums(), 'cross_entropy').result()
            self.client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
            self.consumed_samples += len(self.inputs)

    def evaluate(self):
        if self.protocol == 'twinkle':
            result = self.client.forward_only(self.inputs).result
            values = result['logps']
        else:
            result = self.client.forward(self.datums(), 'cross_entropy').result()
            values = [output['logprobs'].tolist() for output in result.loss_fn_outputs]
        return checked_logps(values)

    def save(self, name):
        if self.protocol == 'twinkle':
            path = self.client.save(
                name=name, save_optimizer=True, consumed_train_samples=self.consumed_samples).twinkle_path
        else:
            path = self.client.save_state(name).result().path
        assert path, 'Checkpoint path is empty'
        return path

    def resume(self, path):
        if self.protocol == 'twinkle':
            progress = self.client.resume_from_checkpoint(path)
            self.consumed_samples = progress['consumed_train_samples']
            return progress
        old = self.identity
        self.client = self.service.create_training_client_from_state_with_optimizer(path=path)
        self.identity = self.client.model_id
        unload_tinker_model(self.service, old)

    def assert_stale(self):
        try:
            if self.protocol == 'twinkle':
                self.client.get_train_configs().result
            else:
                self.client.get_info()
        except Exception as exc:
            assert any(term in str(exc).lower() for term in ('404', 'not found', 'handle', 'active', 'unavailable')), exc
        else:
            raise AssertionError('An old training handle remained usable')

    def close(self, stale=False):
        if self.closed:
            return
        try:
            if not stale:
                if self.protocol == 'twinkle':
                    self.client.remove_adapter()
                else:
                    unload_tinker_model(self.service, self.identity)
        finally:
            if self.transport is not None:
                self.transport.close()
            elif hasattr(self, 'service'):
                self.service.holder.close()
            self.closed = True


def wait_until(predicate, timeout, label):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(label)
        time.sleep(0.5)


def assert_released(controller, namespace, redis_url=None, prefix=None):
    import ray
    from ray import serve
    from ray.util.placement_group import placement_group_table
    diagnostic = ray.get(controller.diagnostics.remote(), timeout=30)
    assert all(row['state'] == 'STOPPED' for row in diagnostic['instances']), diagnostic
    # Training teardown must not require a healthy sampling pool to disappear.
    owned_prefix = 'resources-' + namespace + '-controller_'
    assert not [name for name in serve.status().applications if name.startswith(owned_prefix)]
    assert not [row for row in placement_group_table().values()
                if row.get('name', '').startswith(owned_prefix) and row.get('state') != 'REMOVED']
    if redis_url:
        import redis
        client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
        try:
            assert not list(client.scan_iter(match=prefix + 'model::training_*'))
            assert not list(client.scan_iter(match=prefix + 'seq_ledger::training_*'))
            for key in client.scan_iter(match=prefix + 'token_count::*'):
                assert int(json.loads(client.get(key)) or 0) == 0, 'Quota drift'
        finally:
            client.close()
    return True
