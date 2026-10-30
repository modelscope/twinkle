'Local Qwen3-TTS model loading and adapter configuration.'
import torch

from .data import Qwen3TTSDataConfig

DEFAULT_MODEL_PATH = '/models/Qwen3-TTS-12Hz-1.7B-Base'
# Keep the adapter identifier stable for existing training checkpoints.
ADAPTER = 'smoke'


def data_config(config):
    tc = config.talker_config
    return Qwen3TTSDataConfig(
        **{name: getattr(config, name) for name in
           ('tts_pad_token_id', 'tts_bos_token_id', 'tts_eos_token_id')},
        **{name: getattr(tc, name) for name in
           ('codec_pad_id', 'codec_bos_id', 'codec_eos_token_id',
            'codec_nothink_id', 'codec_think_bos_id', 'codec_think_eos_id',
            'num_code_groups')})


def load_twinkle_model(path, device_mesh):
    # Twinkle constructs a Template (AutoConfig/AutoProcessor) as well as the
    # supplied model_cls. Passing config alone does not register these classes.
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
    from .model import Qwen3TTSModel
    AutoConfig.register('qwen3_tts', Qwen3TTSConfig, exist_ok=True)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)
    model = Qwen3TTSModel(
        model_id=str(path), model_cls=Qwen3TTSForConditionalGeneration,
        config=Qwen3TTSConfig.from_pretrained(path, local_files_only=True), device_mesh=device_mesh,
        mixed_precision='bf16', torch_dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True)
    # A single short sample does not need activation checkpointing. In
    # particular, reentrant checkpointing can sever gradients from frozen inputs.
    model.model.gradient_checkpointing_disable()
    return model
