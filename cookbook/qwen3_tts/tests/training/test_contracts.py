"""CPU loop tests with real autograd/SGD and an explicit Twinkle protocol double."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('training', ROOT / 'tts/qwen3/training.py')
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)


class Bridge:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)
        self.group = types.SimpleNamespace(gradient_accumulation_steps=1, cur_step=0,
                                         train_status=types.SimpleNamespace(num_tokens=0))
        self.optimizer_group = {'test': self.group}
        self.updates = []

    def forward_backward(self, inputs, adapter_name):
        loss = (self.model.weight - inputs['target']).square().mean()
        loss.backward()
        self.group.cur_step += 1
        self.group.train_status.num_tokens += 1
        return {'loss': loss.item()}

    def clip_grad_and_step(self, adapter_name):
        count = self.group.train_status.num_tokens
        for p in self.model.parameters():
            p.grad.div_(count)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.updates.append((self.group.cur_step, count))
        self.group.train_status.num_tokens = 0


def batch(target):
    return {'input_ids': torch.zeros(1, 4, 2), 'target': float(target)}


class TrainingContracts(unittest.TestCase):
    def test_tail_matches_independent_mean_gradient_reference(self):
        model = Bridge()
        progress = training.Progress(2)
        targets = [2, -1, 3, 4, -2]
        reference = torch.tensor(1., requires_grad=True)
        optimizer = torch.optim.SGD([reference], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
        for chunk in [targets[:2], targets[2:4], targets[4:]]:
            ((reference - torch.tensor(chunk)).square().mean()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        for chunk in training.windows([batch(t) for t in targets], 2):
            training.train_window(model, chunk, progress, 'test')
        torch.testing.assert_close(model.model.weight.squeeze(), reference)
        self.assertEqual(model.updates, [(2, 2), (4, 2), (5, 1)])
        self.assertEqual((progress.micro_steps, progress.optimizer_steps, progress.consumed_samples), (5, 3, 5))
        self.assertEqual(progress.pending_micro_steps, 0)
        self.assertEqual(model.scheduler.last_epoch, 3)

    def test_invalid_batch_is_rejected_before_any_backward(self):
        model = Bridge()
        bad = batch(1)
        bad['input_ids'] = torch.zeros(2, 4, 2)
        with self.assertRaisesRegex(ValueError, 'single-sample'):
            training.train_window(model, [batch(1), bad], training.Progress(2), 'test')
        self.assertEqual(model.group.cur_step, 0)
        self.assertIsNone(model.model.weight.grad)

    def test_empty_windows_and_invalid_accumulation_do_not_step(self):
        self.assertEqual(list(training.windows([], 2)), [])
        with self.assertRaises(ValueError):
            training.Progress(0)
        with self.assertRaises(ValueError):
            training.train_window(Bridge(), [], training.Progress(2), 'test')

    def test_pending_gradients_cannot_be_checkpointed(self):
        model = Bridge()
        progress = training.Progress(2, pending_micro_steps=1)
        with self.assertRaisesRegex(ValueError, 'unstepped'):
            training.save_boundary(model, progress, '.', 'unused', 'test', 'sample')
        progress.pending_micro_steps = 0
        model.model.weight.grad = torch.ones_like(model.model.weight)
        with self.assertRaisesRegex(ValueError, 'cleared'):
            training.save_boundary(model, progress, '.', 'unused', 'test', 'sample')

    def test_missing_resume_files_never_silently_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                training.resume_boundary(Bridge(), temp, 'test', 'sample', 2)

    def test_mismatched_resume_metadata_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                         'scheduler.pt', 'rng_state.pt', 'trainer_state.json'):
                (root / name).write_text('{}')
            state = {'format_version': 1, 'sample_fingerprint': 'other',
                     'progress': dict(accumulation_steps=2, micro_steps=2, optimizer_steps=1,
                                      consumed_samples=2, pending_micro_steps=0)}
            (root / 'cookbook_progress.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'sample mismatch'):
                training.resume_boundary(Bridge(), root, 'test', 'sample', 2)
            state['sample_fingerprint'] = 'sample'
            (root / 'cookbook_progress.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'accumulation mismatch'):
                training.resume_boundary(Bridge(), root, 'test', 'sample', 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
