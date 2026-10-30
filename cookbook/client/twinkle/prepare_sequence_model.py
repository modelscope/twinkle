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


def tiny_backbone(directory, seed=42):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, pad_token_id=0, bos_token_id=2, eos_token_id=3,
                             attention_dropout=0., tie_word_embeddings=False)
        Qwen2ForCausalLM(config).save_pretrained(directory)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}
    # Use the model family's tokenizer so AutoTokenizer preserves its byte vocabulary.
    vocab.update({token: index + 4 for index, token in enumerate(sorted(ByteLevel.alphabet()))})
    Qwen2Tokenizer(vocab=vocab, merges=[], pad_token='[PAD]', unk_token='[UNK]',
                   bos_token='[BOS]', eos_token='[EOS]').save_pretrained(directory)
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
    if mcore_native and config.model_type != 'qwen3_5':
        raise ValueError('--mcore-native currently supports Qwen3.5 checkpoints')
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('The tokenizer needs a pad or EOS token')
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
    parser.add_argument('--model', default='ms://Qwen/Qwen3.5-4B')
    parser.add_argument('--output-dir', default='output/cs-models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tiny', action='store_true', help='Build a small offline Qwen2 backbone')
    parser.add_argument('--mcore-native', action='store_true',
                        help='Also write Qwen3.5 Megatron-native text-only checkpoints')
    args = parser.parse_args()
    source = tiny_backbone(Path(args.output_dir) / 'source', args.seed) if args.tiny else args.model
    prepare_models(source, args.output_dir, seed=args.seed, mcore_native=args.mcore_native)


if __name__ == '__main__':
    main()
