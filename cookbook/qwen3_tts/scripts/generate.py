"""Generate one WAV from a local Qwen3-TTS Base model and optional LoRA adapter."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--checkpoint', help='Optional adapter directory; omit for the base model')
    parser.add_argument('--reference_audio', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', required=True, help='New output WAV path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args(argv)
    if not args.text.strip() or args.max_new_tokens < 2:
        parser.error('Nonempty text and max_new_tokens >= 2 required')
    if Path(args.output).suffix.lower() != '.wav':
        parser.error('--output must end in .wav')
    return args


def write_wave(path, wave, sample_rate):
    import numpy as np
    import soundfile as sf
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError('Expected a finite nonempty mono waveform')
    if sample_rate <= 0 or not np.any(wave):
        raise ValueError('Invalid sample rate or silent waveform')
    # Preserve levels without clipping; never overwrite an earlier result.
    with Path(path).open('xb') as output:
        sf.write(output, wave, sample_rate, format='WAV', subtype='FLOAT')


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import random
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import twinkle
    from twinkle import DeviceMesh
    from transformers import AutoProcessor
    from qwen_tts import Qwen3TTSModel as OfficialModel
    from tts.qwen3.runtime import load_twinkle_model, ADAPTER
    from prepare_split_manifest import read_audio

    torch.npu.set_device(0)
    mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
    twinkle.initialize(mode='local', global_device_mesh=mesh)
    path = Path(args.model_path).resolve(strict=True)
    model = load_twinkle_model(path, mesh)
    if args.checkpoint:
        model.add_adapter_to_model(ADAPTER, str(Path(args.checkpoint).resolve(strict=True)), is_trainable=False)
    model._lazy_wrap_model()
    core = model.strategy.unwrap_model(model.model)
    if args.checkpoint:
        core.set_adapter(ADAPTER)
        core = core.get_base_model()
    core.requires_grad_(False)
    core.eval()
    core.speech_tokenizer.model.to(device='cpu', dtype=torch.float32).eval()
    core.speech_tokenizer.device = torch.device('cpu')
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, fix_mistral_regex=True)
    official = OfficialModel(core, processor, generate_defaults=core.generate_config)
    wave, _ = read_audio(Path(args.reference_audio).resolve(strict=True))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    with torch.inference_mode():
        prompt = official.create_voice_clone_prompt(ref_audio=(wave, 24000), x_vector_only_mode=True)
        waves, sample_rate = official.generate_voice_clone(
            text=args.text, language='Auto', voice_clone_prompt=prompt,
            do_sample=True, subtalker_dosample=True,
            non_streaming_mode=True, max_new_tokens=args.max_new_tokens)
    if len(waves) != 1:
        raise ValueError('Expected one generated waveform')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wave(output, waves[0], sample_rate)
    print(f'Wrote {output}; listen for content accuracy and possible truncation.')


if __name__ == '__main__':
    main()
