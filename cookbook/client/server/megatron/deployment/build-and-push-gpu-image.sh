#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/../../../../.."
revision=$(git rev-parse HEAD)
registry=${REGISTRY:-openmind-registry.cn-hangzhou.cr.aliyuncs.com}
image="$registry/openmind/modelscope:twinkle-h20-group-${revision:0:12}"
build_args=()
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    build_args+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
fi

sudo -v
sudo docker login "$registry"
printf 'Building %s from Git commit %s\n' "$image" "$revision"
git archive "$revision" -- .dockerignore Dockerfile pyproject.toml setup.cfg README.md LICENSE src cookbook | \
    sudo env DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --progress=plain \
    "${build_args[@]}" \
    --label "org.opencontainers.image.revision=$revision" -t "$image" -

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
    'sentence-transformers': '6.3.0',
    'transformer-engine': '2.19.0',
    'transformer-engine-cu13': '2.19.0',
    'transformer-engine-torch': '2.19.0',
}
for name in (*expected, 'twinkle-kit', 'ms-swift', 'transformers', 'peft', 'tilelang', 'apache-tvm-ffi'):
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
from sentence_transformers import SentenceTransformer  # noqa: F401
from swift.dev import template_contract  # noqa: F401
from swift.dev.model.multi_lora_sentence_transformer_model import MultiLoraSentenceTransformerModel  # noqa: F401
from transformer_engine.pytorch.optimizers import FusedAdam, multi_tensor_scale  # noqa: F401

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
