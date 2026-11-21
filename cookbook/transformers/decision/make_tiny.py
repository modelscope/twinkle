# Copyright (c) ModelScope Contributors. All rights reserved.
"""Create a random tiny checkpoint and synthetic decisions, with no downloads."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    words = ['[PAD]', '[UNK]', '[EOS]', 'false', 'true'] + list('012345') + list('ABCDEFGHIJKLMNOP')
    tokenizer = Tokenizer(models.WordLevel({s: i for i, s in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
    fast.save_pretrained(output)
    config = LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        pad_token_id=0, eos_token_id=2)
    LlamaForCausalLM(config).save_pretrained(output)
    records = [
        {'state': 'one plus one equals two', 'questions': [{'kind': 'noul', 'question': 'Correct?', 'gold': 1}]},
        {'state': 'cold water', 'questions': [{'kind': 'choice', 'question': 'Temperature?',
                                              'options': ['cold', 'hot'], 'gold': 0}]},
        {'state': 'example rating', 'questions': [{'kind': 'score', 'question': 'Rate it',
                                                  'gold': [.1, .1, .1, .2, .4, .1]}]},
    ]
    (output / 'mini.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))


if __name__ == '__main__':
    main()
