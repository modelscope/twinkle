"""Adapt the production Compose configuration to one network namespace."""

import sys
from pathlib import Path

import json
import yaml
from render_dashboard import render

CONFIG = Path(__file__).resolve().parent


def load(name):
    return yaml.safe_load((CONFIG / name).read_text())


def rebase(value, source, destination):
    if isinstance(value, dict):
        return {key: rebase(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, source, destination) for item in value]
    if isinstance(value, str) and (value == source or value.startswith(source + '/')):
        return destination + value[len(source):]
    return value


def main(output, data, limits, key_prefix):
    for index, (name, old_dir) in enumerate((('mimir', '/data'), ('loki', '/loki'), ('tempo', '/var/tempo'))):
        path = data / name
        path.mkdir(parents=True, exist_ok=True)
        config = rebase(load(f'{name}.yaml'), old_dir, str(path))
        config['server'].update(http_listen_address='127.0.0.1', grpc_listen_port=9095 + index)
        config.setdefault('memberlist', {})['bind_port'] = 7946 + index
        if name == 'tempo':
            config['distributor']['receivers']['otlp']['protocols']['grpc']['endpoint'] = '127.0.0.1:14317'
        (output / f'{name}.yaml').write_text(yaml.safe_dump(config))

    collector = (CONFIG / 'collector.yaml').read_text()
    collector = collector.replace('tempo:4317', '127.0.0.1:14317')
    collector = collector.replace('mimir:9009', '127.0.0.1:9009').replace('loki:3100', '127.0.0.1:3100')
    (output / 'collector.yaml').write_text(collector)

    provisioning = output / 'provisioning'
    (provisioning / 'datasources').mkdir(parents=True)
    (provisioning / 'dashboards').mkdir()
    (output / 'dashboards').mkdir()
    sources = (CONFIG / 'datasources.yaml').read_text()
    for name in ('mimir', 'loki', 'tempo'):
        sources = sources.replace(f'http://{name}:', 'http://127.0.0.1:')
    (provisioning / 'datasources/twinkle.yaml').write_text(sources)
    providers = load('dashboards.yaml')
    providers['providers'][0]['options']['path'] = str(output / 'dashboards')
    (provisioning / 'dashboards/twinkle.yaml').write_text(yaml.safe_dump(providers))
    dashboard = render(json.loads(limits.read_text()), key_prefix)
    (output / 'dashboards/twinkle-multitenant.json').write_text(json.dumps(dashboard) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
