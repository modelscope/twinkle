#!/bin/sh
set -eu
TEMPLATE="${TEMPLATE:-twinkle-code}"
CPU_COUNT="${CPU_COUNT:-1}"
MEMORY_MB="${MEMORY_MB:-1024}"
BASE_IMAGE="${BASE_IMAGE:-}"
# Where the runtime config is copied to, readable by the aenv user. serve.sh
# reads the same default.
REPO_ROOT="${REPO_ROOT:-$HOME/AgentENV}"
CONFIG_DIR="${CONFIG_DIR:-/var/lib/aenv/config}"

SKIP_INSTALL=0
SKIP_BUILD=0
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        # Bootstrap the host but build no template: used by cookbook setups that
        # bring their own Dockerfile and only need the server installed once.
        --skip-build) SKIP_BUILD=1 ;;
        --rebuild) REBUILD=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

if [ "$SKIP_INSTALL" = "0" ]; then
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
fi

echo "==> Authenticating the CLI"
if [ -f "$HOME/.config/aenv/credentials" ]; then
    echo "    already authenticated ($HOME/.config/aenv/credentials)"
else
    aenv auth
fi

if [ "$SKIP_BUILD" = "1" ]; then
    echo "==> Skipping template build (--skip-build)"
    exit 0
fi

if [ "$REBUILD" = "1" ]; then
    echo "==> Deleting template '$TEMPLATE'"
    aenv template delete "$TEMPLATE" || true
fi

echo "==> Building template '$TEMPLATE' (cpu=$CPU_COUNT mem=${MEMORY_MB}MiB)"
set -- Dockerfile -t "$TEMPLATE" --cpu-count "$CPU_COUNT" --memory-mb "$MEMORY_MB"
[ -n "$BASE_IMAGE" ] && set -- "$@" --image "$BASE_IMAGE"
aenv build "$@"

echo
echo "Build runs server-side and takes a few minutes. Follow it with:"
echo "    aenv template watch <template-id>      # id printed above"
echo "    aenv template list                     # confirm it reaches ready"
echo
echo "Then start the server:"
echo "    sh serve.sh"
