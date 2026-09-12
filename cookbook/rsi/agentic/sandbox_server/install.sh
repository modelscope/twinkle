#!/bin/sh
# Install AgentENV and build the sandbox template for agentic RSI.
#
# Usage:
#     sh install.sh                   # install AgentENV, build from Dockerfile
#     sh install.sh --via=sandbox     # build inside a live sandbox instead
#     sh install.sh --skip-install    # template only, AgentENV already installed
#     sh install.sh --rebuild         # delete the old template first
#
# --via picks who runs the install steps, and the only reason to choose is the
# network: the template builder's VM downloaded at 33 KB/s on our host against a
# sandbox's 5.4 MB/s, which turns a six-minute build into one that looks stuck.
#     dockerfile  `aenv build Dockerfile` -- the builder, image config included
#     sandbox     build_via_sandbox.sh -- install in a live sandbox, snapshot it
set -eu

TEMPLATE="${TEMPLATE:-twinkle-rsi-msagent}"
# ms-agent pulls in pandas/matplotlib/modelscope and notebook_executor starts a
# real ipykernel, so 1GiB is not enough.
CPU_COUNT="${CPU_COUNT:-2}"
MEMORY_MB="${MEMORY_MB:-2048}"
# The base image both paths start from. Set this when the host cannot reach
# Docker Hub, e.g. docker.m.daocloud.io/library/python:3.11-slim -- a
# third-party proxy, so prefer your own Aliyun accelerator address
# (<id>.mirror.aliyuncs.com) if you have one. Empty leaves each path on its own
# default: the Dockerfile's `FROM`, or the mirror build_via_sandbox.sh names.
BASE_IMAGE="${BASE_IMAGE:-}"
# Where the runtime config is copied to, readable by the aenv user. serve.sh
# reads the same default.
REPO_ROOT="${REPO_ROOT:-$HOME/AgentENV}"
CONFIG_DIR="${CONFIG_DIR:-/var/lib/aenv/config}"

VIA=dockerfile
SKIP_INSTALL=0
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --via=dockerfile|--via=sandbox) VIA=${arg#--via=} ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --rebuild) REBUILD=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

install_agentenv() {
    echo "==> Installing AgentENV server + aenv CLI"
    curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh \
        | sudo bash

    echo "==> Provisioning the host (kvm group, ublk module, udev, sysctl)"
    sudo server --setup-host
    sudo install -d -o aenv -g aenv /var/lib/aenv/home

    # A source-built binary defaults to its build-time repo path for the config
    # (CARGO_MANIFEST_DIR), which the aenv user cannot read when the repo lives
    # under /root. Only default.toml needs copying — deps_manifest.toml is
    # include_str!'d into the binary at compile time.
    if [ -f "$REPO_ROOT/config/default.toml" ]; then
        sudo install -d -o aenv -g aenv "$CONFIG_DIR"
        sudo install -o aenv -g aenv -m 0644 \
            "$REPO_ROOT/config/default.toml" "$CONFIG_DIR/config.toml"
        echo "    config seeded to $CONFIG_DIR/config.toml"
    fi
}

authenticate() {
    echo "==> Authenticating the CLI"
    if [ -f "$HOME/.config/aenv/credentials" ]; then
        echo "    already authenticated ($HOME/.config/aenv/credentials)"
    else
        aenv auth
    fi
}

build_from_dockerfile() {
    set -- Dockerfile -t "$TEMPLATE" --cpu-count "$CPU_COUNT" --memory-mb "$MEMORY_MB"
    [ -z "$BASE_IMAGE" ] || set -- "$@" --image "$BASE_IMAGE"
    aenv build "$@"
    echo
    echo "Build runs server-side and takes a few minutes. Follow it with:"
    echo "    aenv template watch <template-id>      # id printed above"
    echo "    aenv template list                     # confirm it reaches ready"
}

build_in_sandbox() {
    # DISK_MB and TTL stay that script's business; only what this one owns is
    # handed over, under the names it reads.
    NAME="$TEMPLATE" CPU="$CPU_COUNT" MEMORY_MB="$MEMORY_MB" \
        BASE_IMAGE="$BASE_IMAGE" sh build_via_sandbox.sh
}

[ "$SKIP_INSTALL" = 1 ] || install_agentenv
authenticate

if [ "$REBUILD" = 1 ]; then
    echo "==> Deleting template '$TEMPLATE'"
    aenv template delete "$TEMPLATE" || true
fi

echo "==> Building template '$TEMPLATE' via $VIA (cpu=$CPU_COUNT mem=${MEMORY_MB}MiB)"
if [ "$VIA" = dockerfile ]; then
    build_from_dockerfile
else
    build_in_sandbox
fi

echo
echo "Then start the server:"
echo "    sh serve.sh"
