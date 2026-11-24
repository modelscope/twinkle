# Copyright (c) ModelScope Contributors. All rights reserved.
"""Render the provisioned multitenant dashboard with deployment reference limits."""
import argparse
import math
from pathlib import Path

import json

TEMPLATE = Path(__file__).resolve().parents[1] / 'grafana/dashboards/twinkle-multitenant.json'
LIMITS = ('requests_per_second', 'input_tokens_per_second')


def reference(value: float) -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Usage reference limits must be positive finite numbers')
    return str(value)


def render(config: dict, key_prefix: str) -> dict:
    dashboard = json.loads(TEMPLATE.read_text())
    if set(config) != set(LIMITS):
        raise ValueError(f'Expected reference limits for {", ".join(LIMITS)}')
    limits = {name: reference(value) for name, value in config.items()}
    for panel in dashboard['panels']:
        for target in panel.get('targets', ()):
            if 'keyName' in target:
                target['keyName'] = key_prefix + target['keyName']
            limit = target.pop('referenceLimit', None)
            if limit is not None:
                target['expr'] = f"({target['expr']}) / {limits[limit]}"
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('limits', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--redis-key-prefix', default='', help='Existing Runtime persistence.key_prefix')
    args = parser.parse_args()
    config = json.loads(args.limits.read_text())
    try:
        dashboard = render(config, args.redis_key_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(dashboard, indent=2) + '\n')


if __name__ == '__main__':
    main()
