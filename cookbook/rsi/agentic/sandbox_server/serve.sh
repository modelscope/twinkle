#!/bin/sh
# Start the AgentENV server that hosts the RSI sandboxes.
#
# Usage:
#     sh serve.sh                       # foreground, binds 127.0.0.1:8000
#     API_ADDR=0.0.0.0:8000 sh serve.sh # listen on all interfaces
#     NOHUP=1 sh serve.sh               # background, logs to /tmp/aenv-server.log
#     STOP_ONLY=1 sh serve.sh           # shut down without starting again
set -eu
REPO_ROOT="${REPO_ROOT:-$HOME/AgentENV}"
# Read by the server itself, not by this script.
export API_ADDR="${API_ADDR:-127.0.0.1:8000}"
LOG_FILE="${LOG_FILE:-/tmp/aenv-server.log}"
NOHUP="${NOHUP:-0}"

# The server drops privileges to a non-root user, so it must not inherit root's
# HOME — regctl and docker credential lookups fail with EACCES there, which
# turns into a hard failure once a private registry needs credentials.
AENV_HOME="${AENV_HOME:-/var/lib/aenv/home}"

# The binary bakes in its build-time repo path as the default config location
# (CARGO_MANIFEST_DIR in src/cfg.rs), so a server built under /root looks for
# /root/AgentENV/config/default.toml — unreadable once it drops to the aenv
# user, since /root is 0700. Point it at a copy the runtime user owns.
AENV_CONFIG_PATH="${AENV_CONFIG_PATH:-/var/lib/aenv/config/config.toml}"

# run-with-capabilities.sh is primarily a test wrapper: when these are unset it
# defaults them to /tmp/aenv-test-<uid>/{home,run}. That sends downloaded
# dependencies (kernel, firecracker, overlaybd — hundreds of MB) to a directory
# that /tmp cleanup wipes, so every restart re-downloads them. Pin the real
# state directory instead; home_path in config.toml points at the same place.
AENV_HOME_PATH="${AENV_HOME_PATH:-/var/lib/aenv}"
AENV_RUNTIME_PATH="${AENV_RUNTIME_PATH:-/run/aenv}"

if [ ! -r "$AENV_CONFIG_PATH" ]; then
    echo "Config not readable: $AENV_CONFIG_PATH" >&2
    echo "Seed it from the repo (install.sh does this for you):" >&2
    echo "    sudo install -d -o aenv -g aenv \$(dirname $AENV_CONFIG_PATH)" >&2
    echo "    sudo install -o aenv -g aenv -m 0644 \\" >&2
    echo "        $REPO_ROOT/config/default.toml $AENV_CONFIG_PATH" >&2
    exit 1
fi

# Stop whatever is already running, so this script is a restart rather than a
# "port already in use" failure. Match the binary path, not this script's name:
# run-with-capabilities.sh ends in `exec setpriv ... server`, which replaces the
# process image, so argv[0] of the live process is the server binary.
SERVER_BIN="${SERVER_BIN:-/usr/local/bin/server}"

stop_running() {
    # A systemd-managed instance would be restarted right after a kill, so hand
    # it over to systemctl instead. install.sh sets up aenv.service when systemd
    # is present.
    if [ -d /run/systemd/system ] && systemctl is-active --quiet aenv 2>/dev/null; then
        echo "Stopping systemd service aenv"
        sudo systemctl stop aenv
        return
    fi

    pids=$(pgrep -f "^$SERVER_BIN" 2>/dev/null || true)
    [ -z "$pids" ] && return

    echo "Stopping running server (pid: $pids)"
    # SIGTERM first: the server tears down microVMs, veth pairs and iptables
    # rules on shutdown, and SIGKILL would leave those behind.
    sudo kill $pids 2>/dev/null || true
    i=0
    while [ $i -lt 30 ] && pgrep -f "^$SERVER_BIN" >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
    done
    if pgrep -f "^$SERVER_BIN" >/dev/null 2>&1; then
        echo "  still alive after 30s, sending SIGKILL"
        sudo pkill -KILL -f "^$SERVER_BIN" 2>/dev/null || true
        sleep 1
    fi
}

stop_running

if [ "${STOP_ONLY:-0}" = "1" ]; then
    echo "Stopped."
    exit 0
fi

cd "$REPO_ROOT"

# run-with-capabilities.sh grants CAP_NET_ADMIN + CAP_SYS_ADMIN via setpriv and
# re-initialises supplementary groups (--init-groups), which is what makes a
# fresh kvm-group membership take effect without re-login. It derives repo_root
# from BASH_SOURCE, so the path above is what matters, not the cwd.
#
# `sudo env VAR=...`, not `sudo VAR=...`: with sudoers env_reset (the default)
# the latter is not guaranteed to pass anything through.
#
# AENV_RUN_USER must be explicit: the script otherwise falls back through
# SUDO_USER -> repo owner -> aenv -> root, and running as root is not supported.
E="AENV_RUN_USER=aenv HOME=$AENV_HOME API_ADDR=$API_ADDR AENV_CONFIG_PATH=$AENV_CONFIG_PATH AENV_HOME_PATH=$AENV_HOME_PATH AENV_RUNTIME_PATH=$AENV_RUNTIME_PATH"

if [ "$NOHUP" = "1" ]; then
    # setsid, not just nohup: the wrapper ends in `exec setpriv`, which replaces
    # the process image, and a SIGHUP disposition inherited from nohup is not
    # guaranteed to survive that. A new session detaches from the terminal
    # regardless.
    echo "Starting AgentENV on $API_ADDR (background) -> $LOG_FILE"
    sudo env $E setsid nohup ./scripts/run-with-capabilities.sh server \
        >"$LOG_FILE" 2>&1 </dev/null &
    echo "Tail with: tail -f $LOG_FILE"
else
    echo "Starting AgentENV on $API_ADDR (foreground, Ctrl-C to stop)"
    exec sudo env $E ./scripts/run-with-capabilities.sh server
fi
