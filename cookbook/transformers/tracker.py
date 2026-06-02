import os
from peft import LoraConfig
import twinkle
from twinkle import get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from twinkle.tracker import register_tracker, list_trackers, clear_trackers
logger = get_logger()
# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID = 'ms://Qwen/Qwen2.5-0.5B-Instruct'
DATASET_ID = 'ms://swift/self-cognition'
TEMPLATE_NAME = 'Template'
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
TRAIN_STEPS = 5
# ── Tracker selection ──────────────────────────────────────────────────────────
def setup_tracker():
    """Register either SwanLabTracker (if API key available) or PrintTracker."""
    if os.environ.get('SWANLAB_API_KEY'):
        from twinkle.tracker.swanlab import SwanLabTracker
        tracker = SwanLabTracker(
            project='twinkle-test',
            experiment_name='tracker-integration-test',
            config={'model': MODEL_ID, 'lr': LEARNING_RATE, 'steps': TRAIN_STEPS},
            output_dir='./test_tracker_output',
        )
        register_tracker(tracker)
        logger.info('SwanLabTracker registered — project=twinkle-test')
        return tracker
    else:
        from twinkle.tracker import ExperimentTracker
        class PrintTracker(ExperimentTracker):
            def __init__(self):
                self.logged: list[tuple[int, dict]] = []
            def log(self, data: dict, step: int) -> None:
                self.logged.append((step, data))
                logger.info('[PrintTracker] step=%s metrics=%s', step, data)
            def cleanup(self) -> None:
                logger.info('[PrintTracker] cleanup — %s dispatches', len(self.logged))
        tracker = PrintTracker()
        register_tracker(tracker)
        logger.info('PrintTracker registered (set SWANLAB_API_KEY for SwanLab)')
        return tracker
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    twinkle.initialize(mode='local', seed=42)
    tracker = setup_tracker()
    assert len(list_trackers()) == 1
    logger.info('Tracker ready: %s', type(tracker).__name__)
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(10)))
    dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('test_model', 'test_author'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    model = TransformersModel(model_id=MODEL_ID)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=1, num_training_steps=TRAIN_STEPS
    )
    for step, batch in enumerate(dataloader):
        if step >= TRAIN_STEPS:
            break
        model.forward_backward(inputs=batch)
        model.clip_grad_and_step()
        metric = model.calculate_metric(is_training=True)
        logger.info('Step %s raw metric: %s', step + 1, metric)
    # Verification (only works for PrintTracker)
    if hasattr(tracker, 'logged'):
        n = len(tracker.logged)
        assert n > 0, 'No metrics were dispatched — dispatch() not called'
        logger.info('=== Dispatch verification ===')
        logger.info('Total dispatches: %s', n)
        for i, (step, data) in enumerate(tracker.logged):
            all_floats = all(isinstance(v, float) for v in data.values())
            logger.info('  [%s] step=%s keys=%s all_float=%s', i + 1, step, list(data.keys()), all_floats)
    clear_trackers()
    assert len(list_trackers()) == 0
    logger.info('=== Test complete ===')
if __name__ == '__main__':
    main()
