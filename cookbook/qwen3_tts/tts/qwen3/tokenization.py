"""One local-only tokenizer policy for training, templates and adapter reload."""

def load_text_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, fix_mistral_regex=True)


def local_template(path):
    from twinkle.template import Template
    return Template(str(path), local_files_only=True, fix_mistral_regex=True)


def assert_tokenizers_match(reference, others, texts):
    """Compare actual IDs, and ensure the reference preserves input text."""
    for text in texts:
        ids = reference.encode(text, add_special_tokens=False)
        if reference.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
            raise AssertionError('Tokenizer does not preserve probe text')
        for tokenizer in others:
            if tokenizer.encode(text, add_special_tokens=False) != ids:
                raise AssertionError('Tokenizer IDs differ across training/template/reload')
