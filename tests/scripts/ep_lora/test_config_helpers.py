from types import SimpleNamespace

from ep_lora_config_helpers import get_text_config, get_vocab_size, set_text_config_attrs


def test_get_vocab_size_prefers_nested_text_config():
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=248320))

    assert get_vocab_size(config) == 248320


def test_set_text_config_attrs_updates_nested_text_config():
    text_config = SimpleNamespace(num_hidden_layers=40, use_cache=True)
    config = SimpleNamespace(text_config=text_config)

    set_text_config_attrs(config, num_hidden_layers=2, use_cache=False)

    assert get_text_config(config).num_hidden_layers == 2
    assert get_text_config(config).use_cache is False


def test_set_text_config_attrs_falls_back_to_top_level_config():
    config = SimpleNamespace(vocab_size=1024)

    set_text_config_attrs(config, num_hidden_layers=2, use_cache=False)

    assert config.num_hidden_layers == 2
    assert config.use_cache is False
