#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/../../../../.."
if [[ -n $(git status --porcelain) ]]; then
    printf 'Build from a clean deployment checkout.\n' >&2
    exit 2
fi
revision=$(git rev-parse HEAD)
registry=${REGISTRY:-openmind-registry.cn-hangzhou.cr.aliyuncs.com}
image="$registry/openmind/modelscope:twinkle-h20-group-${revision:0:12}"

sudo -v
sudo docker login "$registry"
sudo env DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --progress=plain \
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" \
    --label "org.opencontainers.image.revision=$revision" -t "$image" .

sudo docker run --rm -i --platform linux/amd64 --entrypoint python "$image" - <<'PYTHON'
import json
from importlib import metadata
import re
import subprocess
import sys

expected = {
    'modelscope': '1.40.3',
    'torch': '2.14.0',
    'vllm': '0.33.0',
    'ray': '2.55.1',
    'megatron-core': '0.20.0',
    'mcore-bridge': '1.7.0',
}
for name in (*expected, 'twinkle', 'ms-swift', 'transformers', 'peft', 'tilelang', 'apache-tvm-ffi'):
    version = metadata.version(name)
    print(f'{name}=={version}', flush=True)
    if name in expected and version.split('+', 1)[0] != expected[name]:
        raise SystemExit(f'Expected {name}=={expected[name]}, got {version}')

swift = metadata.distribution('ms-swift')
origin = json.loads(swift.read_text('direct_url.json') or '{}')
commit = 'e1ac89d24f47cb8fd73a7245dd79df0959aec284'
if origin.get('url') != f'https://codeload.github.com/modelscope/ms-swift/tar.gz/{commit}':
    raise SystemExit('The installed Swift revision does not match the runtime pin.')
print(f'ms-swift revision {commit}', flush=True)
from swift.dev import template_contract  # noqa: F401

check = subprocess.run(
    [sys.executable, '-m', 'pip', 'check'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(check.stdout, end='', flush=True)
if check.returncode:
    known = re.compile(
        r'(?:gradio 5\.32\.1 has requirement (?:pillow|pydantic|starlette)[^,]*.*, but you have .*'
        r'|decord 0\.6\.0 is not supported on this platform)'
    )
    issues = check.stdout.splitlines()
    if check.returncode != 1 or not issues or not all(known.fullmatch(line) for line in issues):
        raise SystemExit(check.returncode)
    print('Only the known Gradio/Decord base-image exceptions were reported.', flush=True)
PYTHON

sudo docker push "$image"
printf '\nPublished image\n%s\n' "$image"
