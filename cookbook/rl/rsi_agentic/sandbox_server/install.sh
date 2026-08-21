#!/bin/sh
# Build the sandbox template for agentic RSI.
#
# The AgentENV server itself is the same one cookbook/rl/envs uses; only the
# template differs. Its bootstrap (install, host provisioning, config seeding)
# is delegated rather than copied, so there is one place to fix when it changes.
set -eu

TEMPLATE="${TEMPLATE:-twinkle-rsi-msagent}"
# ms-agent pulls in pandas/matplotlib/modelscope and notebook_executor starts a
# real ipykernel, so 1GiB (the plain code template's size) is not enough.
CPU_COUNT="${CPU_COUNT:-2}"
MEMORY_MB="${MEMORY_MB:-2048}"
BASE_IMAGE="${BASE_IMAGE:-}"

SKIP_SERVER=0
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-server) SKIP_SERVER=1 ;;
        --rebuild) REBUILD=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"
REPO_ROOT=$(cd ../../../.. && pwd)
MS_AGENT="$REPO_ROOT/ms-agent"

[ -f "$MS_AGENT/setup.py" ] || {
    echo "ms-agent checkout not found at $MS_AGENT" >&2
    echo "The image installs the same source the training host imports; without it" >&2
    echo "the sandbox would run a different ms-agent than training assumes." >&2
    exit 1
}

if [ "$SKIP_SERVER" = "0" ]; then
    echo "==> Installing the AgentENV server (shared with cookbook/rl/envs)"
    sh ../../envs/agentenv_server/install.sh --skip-build
fi

# A fresh staging directory per run, so a stale ms-agent copy can never end up
# in the image and nothing has to be deleted to make room.
STAGE=$(mktemp -d)
echo "==> Staging build context in $STAGE"
cp Dockerfile "$STAGE/"
# .git and caches are megabytes of noise in a build context and would also
# invalidate the layer cache on every commit.
tar -C "$(dirname "$MS_AGENT")" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    -cf - "$(basename "$MS_AGENT")" | tar -C "$STAGE" -xf -

HEAD_SHA=$(git -C "$MS_AGENT" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "    ms-agent staged at commit $HEAD_SHA"
echo "    the training host must import this same checkout; if it does not,"
echo "    tool behaviour differs between rollout and the agent you deploy."

if [ "$REBUILD" = "1" ]; then
    echo "==> Deleting template '$TEMPLATE'"
    aenv template delete "$TEMPLATE" || true
fi

echo "==> Building template '$TEMPLATE' (cpu=$CPU_COUNT mem=${MEMORY_MB}MiB)"
set -- "$STAGE/Dockerfile" -t "$TEMPLATE" --cpu-count "$CPU_COUNT" --memory-mb "$MEMORY_MB"
[ -n "$BASE_IMAGE" ] && set -- "$@" --image "$BASE_IMAGE"
aenv build "$@"

echo
echo "Build runs server-side and takes a few minutes. Follow it with:"
echo "    aenv template watch <template-id>      # id printed above"
echo "    aenv template list                     # confirm it reaches ready"
echo
echo "Then start the server:"
echo "    sh serve.sh"
