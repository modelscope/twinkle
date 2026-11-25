#!/usr/bin/env bash
set -euo pipefail

role=all
case "${1:-}" in
    all|gateway|observability)
        role=$1
        shift
        ;;
esac
if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [all|gateway|observability] [version]\n' "${0##*/}" >&2
    exit 2
fi

version=${1:-$(date -u +%Y%m%d-%H%M%S)}
if [[ ! "$version" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,109}$ ]]; then
    printf 'Invalid image version: %s\n' "$version" >&2
    exit 2
fi

gateway_dockerfile=cookbook/client/server/megatron/deployment/Dockerfile.gateway
observability_dockerfile=cookbook/observability/Dockerfile.runtime
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
context=$PWD
if [[ -f "$script_dir/Dockerfile.gateway" ]]; then
    context=$(cd -- "$script_dir/../../../../.." && pwd)
elif [[ -f "$script_dir/$gateway_dockerfile" ]]; then
    context=$script_dir
fi
if [[ ! -f "$context/$gateway_dockerfile" || ! -f "$context/$observability_dockerfile" ]]; then
    printf 'Run from the extracted build context or the Twinkle repository.\n' >&2
    exit 2
fi

registry=openmind-registry.cn-hangzhou.cr.aliyuncs.com
alpine_mirror=${ALPINE_MIRROR-https://mirrors.aliyun.com/alpine}
dockerhub_mirror=${DOCKERHUB_MIRROR:-docker.io}
dockerhub_mirror=${dockerhub_mirror#https://}

sudo -v
sudo docker info >/dev/null
sudo docker login "$registry"

roles=("$role")
if [[ "$role" == all ]]; then
    roles=(gateway observability)
fi
images=()
for component in "${roles[@]}"; do
    case "$component" in
        gateway) dockerfile=$gateway_dockerfile ;;
        observability) dockerfile=$observability_dockerfile ;;
    esac
    image="$registry/openmind/modelscope:twinkle-$component-$version"
    printf '\nBuilding %s\n' "$image"
    sudo env DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --progress=plain \
        --build-arg "ALPINE_MIRROR=$alpine_mirror" \
        --build-arg "DOCKERHUB_PREFIX=${dockerhub_mirror%/}/" \
        -f "$context/$dockerfile" -t "$image" "$context"
    images+=("$image")
done
for image in "${images[@]}"; do
    sudo docker push "$image"
done
printf '\nPublished images\n%s\n' "${images[@]}"
