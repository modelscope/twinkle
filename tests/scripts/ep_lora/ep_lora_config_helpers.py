def get_text_config(config):
    return getattr(config, 'text_config', config)


def get_vocab_size(config):
    return get_text_config(config).vocab_size


def set_text_config_attrs(config, **attrs):
    text_config = get_text_config(config)
    for name, value in attrs.items():
        setattr(text_config, name, value)
