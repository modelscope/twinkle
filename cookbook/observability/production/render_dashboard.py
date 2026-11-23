# Copyright (c) ModelScope Contributors. All rights reserved.
"""Render the provisioned multitenant dashboard with deployment reference limits."""
import argparse
import math
import re
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
    overrides = config.get('overrides', {})
    if any(not re.fullmatch('[0-9a-f]{64}', user_id) for user_id in overrides):
        raise ValueError('Usage overrides must use complete Token fingerprints')
    for panel in dashboard['panels']:
        for target in panel['targets']:
            if 'keyName' in target:
                target['keyName'] = key_prefix + target['keyName']
            limit = target.pop('referenceLimit', None)
            if limit is None:
                continue
            usage_rate = target['expr']
            default = reference(config['defaults'][limit])
            specific = ' or '.join(
                f'label_replace(vector({reference(values[limit])}), "user_id", "{user_id}", "__name__", ".*")'
                for user_id, values in overrides.items() if limit in values
            )
            fallback = f'({usage_rate}) / {default}'
            target['expr'] = f'(({usage_rate}) / on (user_id) ({specific})) or ({fallback})' if specific else fallback
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('limits', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--redis-key-prefix', default='', help='Existing Runtime persistence.key_prefix')
    args = parser.parse_args()
    config = json.loads(args.limits.read_text())
    if config['defaults'].keys() != set(LIMITS) or any(
            values.keys() - set(LIMITS) for values in config.get('overrides', {}).values()):
        parser.error(f'Expected reference limits for {", ".join(LIMITS)}')
    args.output.write_text(json.dumps(render(config, args.redis_key_prefix), indent=2) + '\n')


if __name__ == '__main__':
    main()
