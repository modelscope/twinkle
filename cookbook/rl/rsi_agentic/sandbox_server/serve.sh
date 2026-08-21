#!/bin/sh
# Start the AgentENV server that hosts the RSI sandboxes.
#
# Deliberately a delegation, not a copy. It is the same server as
# cookbook/rl/envs uses -- RSI only changes which template the sandboxes boot
# from -- and that script carries a hundred lines of host-provisioning detail
# (capability wrapper, config path, systemd handover) that would silently drift
# if it existed twice.
#
# All of its environment variables still apply, e.g.:
#     API_ADDR=0.0.0.0:8000 NOHUP=1 sh serve.sh
set -eu
cd "$(dirname "$0")"

SHARED=../../envs/agentenv_server/serve.sh
[ -f "$SHARED" ] || {
    echo "Shared AgentENV launcher not found: $SHARED" >&2
    exit 1
}
exec sh "$SHARED" "$@"
