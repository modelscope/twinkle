# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Template utility functions for model and sampler handlers.

Provides centralized template selection logic for different model types,
making it easy to maintain and extend template configurations.
"""

# Template mapping for different model families
# Key: model name pattern to match, Value: template name
MODEL_TEMPLATE_MAPPING = {
    'Qwen3.5': 'Qwen3_5Template',
    'Qwen3.6': 'Qwen3_5Template',
    # Add more model-template mappings here as needed
    # 'ModelName': 'TemplateName',
}

# Default template for models not in the mapping
DEFAULT_TEMPLATE = 'Template'


def get_template_for_model(model_name: str) -> str:
    """
    Get the appropriate template name for a given model.

    This function determines which template to use based on the model name.
    It checks if the model name matches any known patterns and returns the
    corresponding template, or falls back to the default template.

    Args:
        model_name: The name or identifier of the model (e.g., 'Qwen3.5-4B')

    Returns:
        The template name to use (e.g., 'Qwen3_5Template' or 'Template')

    Examples:
        >>> get_template_for_model('Qwen3.5-4B')
        'Qwen3_5Template'
        >>> get_template_for_model('Qwen2-7B')
        'Template'
        >>> get_template_for_model('llama-3-8b')
        'Template'
    """
    for pattern, template in MODEL_TEMPLATE_MAPPING.items():
        if pattern in model_name:
            return template
    return DEFAULT_TEMPLATE
