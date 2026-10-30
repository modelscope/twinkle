# Copyright (c) ModelScope Contributors. All rights reserved.
"""Prepare fixed text backbones and classification heads for CS training.

    python cookbook/client/twinkle/prepare_sequence_model.py --model ms://Qwen/Qwen3.5-4B
    python cookbook/client/twinkle/prepare_sequence_model.py --model ms://Qwen/Qwen3.5-4B --mcore-native
    python cookbook/client/twinkle/prepare_sequence_model.py --tiny

Checkpoints land in output/cs-models/{embedding,reranker,seq_cls}. Cold workers
load these complete bases, never a newly randomized classification head.
--mcore-native also writes output/cs-models-mcore/* with Qwen3.5's original
configuration and model.language_model.* keys for Megatron text-only training.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def tiny_backbone(directory, tokenizer_source='ms://Qwen/Qwen3.5-4B', seed=42):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from twinkle.hub import HubOperation

    tokenizer_path = HubOperation.download_model(tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
                             bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                             attention_dropout=0., tie_word_embeddings=False)
        Qwen2ForCausalLM(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def _mcore_native_key(key):
    if key.startswith('model.') and not key.startswith('model.language_model.'):
        return 'model.language_model.' + key[len('model.'):]
    return key


def _save_mcore_native(source_dir, native_dir, config, width):
    from safetensors.torch import load_file, save_file

    source_dir, native_dir = Path(source_dir), Path(native_dir)
    shutil.copytree(source_dir, native_dir, dirs_exist_ok=True)
    for file in native_dir.glob('model*.safetensors'):
        state = load_file(str(file))
        save_file({_mcore_native_key(key): value for key, value in state.items()}, str(file))
    index_path = native_dir / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        index['weight_map'] = {_mcore_native_key(key): value for key, value in index['weight_map'].items()}
        index_path.write_text(json.dumps(index, indent=2))
    native_config = type(config).from_dict(config.to_dict())
    if width is not None:
        native_config.architectures = ['Qwen3_5ForSequenceClassification']
        native_config.num_labels = width
        native_config.problem_type = 'regression' if width == 1 else 'single_label_classification'
        native_config.tie_word_embeddings = False
        native_config.text_config.tie_word_embeddings = False
    native_config.language_model_only = True
    native_config.save_pretrained(native_dir)


def prepare_models(model_id, output_dir, *, seed=42, mcore_native=False):
    from twinkle.hub import HubOperation

    directory = Path(output_dir)
    source = HubOperation.download_model(str(model_id))
    config = AutoConfig.from_pretrained(source, trust_remote_code=False)
    if config.model_type == 'qwen3_5':
        from packaging.version import Version
        if Version(transformers.__version__) < Version('5.8'):
            raise RuntimeError('Qwen3.5 preparation requires transformers>=5.8; use --tiny or upgrade')
    if mcore_native and config.model_type != 'qwen3_5':
        raise ValueError('--mcore-native currently supports Qwen3.5 checkpoints')
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('The tokenizer needs a pad or EOS token')
    task_directories = [directory / task for task in ('embedding', 'reranker', 'seq_cls')]
    if mcore_native:
        task_directories += [Path(str(directory) + '-mcore')]
    for path in task_directories:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f'Output directory is not empty: {path}; choose a new directory or clear it first')
    base = getattr(transformers, config.architectures[0]).from_pretrained(source, torch_dtype='auto')
    backbone = base.model
    text_config = getattr(config, 'text_config', config)
    if hasattr(backbone, 'language_model'):
        backbone = backbone.language_model
    text_config.pad_token_id = tokenizer.pad_token_id
    native_directory = Path(str(directory) + '-mcore')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        causal = transformers.AutoModelForCausalLM.from_config(text_config)
    causal.to(next(backbone.parameters()).dtype)
    causal.model.load_state_dict(backbone.state_dict(), strict=True)
    causal.lm_head.load_state_dict(base.lm_head.state_dict(), strict=True)
    causal.save_pretrained(directory / 'embedding')
    tokenizer.save_pretrained(directory / 'embedding')
    if mcore_native:
        _save_mcore_native(directory / 'embedding', native_directory / 'embedding', config, None)
    del causal
    for task, width in [('reranker', 1), ('seq_cls', 3)]:
        head_config = type(text_config).from_dict(text_config.to_dict())
        head_config.num_labels = width
        head_config.problem_type = 'regression' if width == 1 else 'single_label_classification'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = AutoModelForSequenceClassification.from_config(head_config)
        model.to(next(backbone.parameters()).dtype)
        model.model.load_state_dict(backbone.state_dict(), strict=True)
        model.save_pretrained(directory / task)
        tokenizer.save_pretrained(directory / task)
        if mcore_native:
            _save_mcore_native(directory / task, native_directory / task, config, width)
        del model
    return {task: str((directory / task).resolve()) for task in ('embedding', 'reranker', 'seq_cls')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help='Source model; omit it to build a tiny model with the Qwen3.5 tokenizer')
    parser.add_argument('--output-dir', default='output/cs-models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tiny', action='store_true', help='Build a small Qwen2 backbone with the source tokenizer')
    parser.add_argument('--mcore-native', action='store_true',
                        help='Also write Qwen3.5 Megatron-native text-only checkpoints')
    args = parser.parse_args()
    tokenizer_source = args.model or 'ms://Qwen/Qwen3.5-4B'
    source = tiny_backbone(Path(args.output_dir) / 'source', tokenizer_source, args.seed) \
        if args.tiny or args.model is None else args.model
    prepare_models(source, args.output_dir, seed=args.seed, mcore_native=args.mcore_native)


if __name__ == '__main__':
    main()
