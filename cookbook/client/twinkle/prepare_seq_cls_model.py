# Twinkle Client - Prepare a base model with a classification head
#
# Reranker and sequence-classification training put LoRA on a FIXED score head.
# The head must be saved with the base model, so every server worker (including
# one restarted later to resume a checkpoint) loads exactly the same head.
# This script adds a randomly initialized ``num_labels``-wide head to the text
# backbone of a causal LM and saves the result, together with the tokenizer.
#
# Run it once on the server machine, from the repository root:
#   python cookbook/client/twinkle/prepare_seq_cls_model.py --num-labels 1 --output-dir output/Qwen3.5-4B-reranker
#   python cookbook/client/twinkle/prepare_seq_cls_model.py --num-labels 3 --output-dir output/Qwen3.5-4B-seq-cls
#
# The output directories are the model_id values used in
# cookbook/client/server/transformer/server_config_seq_cls.yaml.

import argparse
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from twinkle.hub import HubOperation

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ms://Qwen/Qwen3.5-4B')
parser.add_argument('--num-labels', type=int, required=True, help='1 for a reranker score, N for N classes')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()

model_dir = HubOperation.download_model(args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Qwen3.5 is a vision-language checkpoint; the classifier is built on its text part only.
config = AutoConfig.from_pretrained(model_dir)
text_config = getattr(config, 'text_config', config)
text_config.num_labels = args.num_labels
# The classifier scores the last non-padding token; batched inputs need the pad id to find it.
text_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# The load report lists the vision weights as UNEXPECTED and score.weight as MISSING.
model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=text_config, torch_dtype='auto')
# 显式初始化新分类头：不能依赖缺失权重加载时的隐式初始化。
torch.manual_seed(42)
torch.nn.init.normal_(model.score.weight, mean=0.0, std=text_config.initializer_range)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f'Saved {type(model).__name__} with {args.num_labels} label(s) to {args.output_dir}')  # noqa: T201
