import os
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoConfig
from transformers import (
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4TextConfig,
    Gemma4VideoProcessor,
    Gemma4VisionConfig,
    GemmaTokenizer,
    GenerationConfig,
    RopeParameters,
)

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor, LatexOCRProcessor

logger = get_logger()

########## Construct a device_mesh ##########
device_mesh = DeviceMesh.from_sizes(
    # fsdp_size=2,
    # dp_size=1,
    # ep_size=2,
    device_type=Platform.get_platform().device_prefix(),
)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

########## 超参数 ##########
IGNORE_MISMATCHED_SIZES = True
# MODEL_PATH = '/nas/disk1/MiniMax-M2.5'
# MODEL_PATH = '/nas/disk1/Qwen3-30B-A3B'
# MODEL_PATH = '/nas/disk1/gemma-4-E2B-it'
# MODEL_PATH = '/nas/disk1/gemma-4-26B-A4B'
MODEL_PATH = r"C:\Users\wliuu\.cache\modelscope\hub\models\google\gemma-4-E2B-it"

# DATASET_PATH = '/model/lzh/train/datasets/self-cognition.jsonl'
DATASET_PATH = 'ms://AI-ModelScope/LaTeX_OCR'
# DATASET_PATH = r'C:\Users\wliuu\.cache\modelscope\hub\datasets\LaTex_OCR_train.json'

### 注意: gemma4-26b-a4b: text layers=30, vision layers=27
TEXT_NUM_LAYERS = 3
VISION_NUM_LAYERS = 3

TRAIN_LEN = 200
BATCH_SIZE = 4

from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Message, Trajectory
class LatexOCRProcessor(Preprocessor):

    def __call__(self, rows):   # 输入的rows 是pyarrow.Table处理后的inputs
        rows = self.map_col_to_row(rows)    # 构建出一行一行的数据
        rows = [self.preprocess(row) for row in rows]   # 每行构建为Trajectory
        col = self.map_row_to_col(rows)    # 变成列的形式, col['messages']下有BATCH_LEN条 Trajectory数据
        return col

    def preprocess(self, row) -> Trajectory:
        return Trajectory(  # <class 'twinkle.data_format.trajectory.Trajectory'>
            messages=[
                Message(role='user', content='<image>Using LaTeX to perform OCR on the image.', images=[row['image']]),
                Message(role='assistant', content=row['text']), # 相当于有监督学习的label
            ]
        )

def eval(model, eval_dataloader):
    for step, batch in tqdm(enumerate(eval_dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics

def train():

    ### prepare dataset and dataloader
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_PATH, data_slice=range(TRAIN_LEN)))
    # Set template to prepare encoding
    # dataset.set_template('Template', model_id=MODEL_PATH)         # 指定的是Template实例名称
    dataset.set_template('Gemma4Template', model_id=MODEL_PATH)     # 
    # Preprocess the dataset to standard format
    # dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.map(preprocess_func=LatexOCRProcessor)
    # Encode dataset
    dataset.encode()    # 2B可以用Template，26B会报错
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    
    # config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config, kwargs = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        return_unused_kwargs=True,
        # code_revision=code_revision,
        # _commit_hash=commit_hash,
        # **hub_kwargs,
        # **kwargs,
    )

    if isinstance(config, Gemma4Config):    # 减层
        text_config = config.text_config
        vision_config = config.vision_config
        if TEXT_NUM_LAYERS is not None and hasattr(text_config, 'num_hidden_layers'):
            text_config.num_hidden_layers = TEXT_NUM_LAYERS
            print(f" modify > text_config.num_hidden_layers = {text_config.num_hidden_layers}")
        if VISION_NUM_LAYERS is not None and hasattr(vision_config, 'num_hidden_layers'):
            vision_config.num_hidden_layers = VISION_NUM_LAYERS
            print(f" modify > vision_config.num_hidden_layers = {vision_config.num_hidden_layers}")
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    # Use a TransformersModel
    model = TransformersModel(
        model_id=MODEL_PATH,
        config=config,
        device_mesh=device_mesh,
        strategy="accelerate", # native_fsdp、 accelerate
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        fsdp_config={
            'reshard_after_forward': True,
            'expert_parallel': {
                'enabled': True,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            }
        },
    )   # 会直接load weights
    # type(model): <class 'twinkle.model.transformers.transformers.TransformersModel'>
    # type(model.model): <class 'transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration'>
    # 若未传入config, 则自动通过 AutoConfig.from_pretrained 读取config

    # FSDP 不进行切分：
    # model.model._no_split_modules = {'Gemma4VisionEncoderLayer', 'Gemma4TextDecoderLayer', 'Gemma4AudioLayer'}    # for 3 modalities(2B、4B)
    model.model._no_split_modules = {'Gemma4VisionEncoderLayer', 'Gemma4TextDecoderLayer'}    # for 2 modalities(26B-A4B、31B)

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)   # gradient_accumulation_steps？
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)

    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))

    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    best_eval_loss = 99.0
    # lora: 8G * 8
    # full: 18G * 8

    ### eval dataset and dataloader
    # EVAL_LENGTH = 100
    # eval_dataset = Dataset(dataset_meta=DatasetMeta(DATASET_PATH, data_slice=range(EVAL_LENGTH)))
    # eval_dataset.set_template('Gemma4Template', model_id=MODEL_PATH)
    # eval_dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # eval_dataset.encode()
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=8)
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()

        if step % 10 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True) # 若npu显存不足,
            logger.info(f'Current is step {step} of {len(dataloader)}, Train metric: {metric}')

        # if step % 10 == 0:
        #     metrics = eval(model, eval_dataloader)
        #     metrics['step'] = step
        #     if float(metrics['loss']) < best_eval_loss:
        #         # model.save(f'checkpoint-{step}')
        #         best_eval_loss = float(metrics['loss'])
        #     metrics['best_eval_loss'] = best_eval_loss
        #     logger.info(f'Current is step {step} of {len(dataloader)}, Eval metric: {metrics}')
            
    # model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()

