#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [version]\n' "${0##*/}" >&2
    exit 2
fi

version=${1:-$(date -u +%Y%m%d-%H%M%S)}
if [[ ! "$version" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,109}$ ]]; then
    printf 'Invalid image version: %s\n' "$version" >&2
    exit 2
fi

cpu1_dockerfile=cookbook/client/server/megatron/deployment/Dockerfile.cpu1
cpu2_dockerfile=cookbook/observability/Dockerfile.runtime
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
context=$PWD
if [[ -f "$script_dir/Dockerfile.cpu1" ]]; then
    context=$(cd -- "$script_dir/../../../../.." && pwd)
elif [[ -f "$script_dir/$cpu1_dockerfile" ]]; then
    context=$script_dir
fi
if [[ ! -f "$context/$cpu1_dockerfile" || ! -f "$context/$cpu2_dockerfile" ]]; then
    printf 'Run from the extracted build context or the Twinkle repository.\n' >&2
    exit 2
fi

registry=openmind-registry.cn-hangzhou.cr.aliyuncs.com
cpu1_image="$registry/openmind/modelscope:twinkle-cpu1-$version"
cpu2_image="$registry/openmind/modelscope:twinkle-cpu2-$version"
alpine_mirror=${ALPINE_MIRROR-https://mirrors.aliyun.com/alpine}

sudo -v
sudo docker info >/dev/null
sudo docker login "$registry"

build_image() {
    sudo env DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
        --build-arg "ALPINE_MIRROR=$alpine_mirror" \
        -f "$context/$1" -t "$2" "$context"
}

build_image "$cpu1_dockerfile" "$cpu1_image"
build_image "$cpu2_dockerfile" "$cpu2_image"
sudo docker push "$cpu1_image"
sudo docker push "$cpu2_image"
printf '\nPublished images\n%s\n%s\n' "$cpu1_image" "$cpu2_image"
