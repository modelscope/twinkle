#!/usr/bin/env bash
set -euo pipefail
umask 077

root=/opt/twinkle-cpu
source "$root/twinkle-runtime-common.sh"
load_config "${1:-/config/twinkle-cpu1.env}"

: "${TWINKLE_UPSTREAM:?Set the private Serve address, including its port}"
: "${TWINKLE_REDIS_DATA_DIR:?Set the persistent Redis data directory}"
: "${TWINKLE_REDIS_PASSWORD:?Set the Redis runtime password}"
: "${OBS_REDIS_PASSWORD:?Set the Redis Grafana password}"
mkdir -p -- "$TWINKLE_REDIS_DATA_DIR"
work_dir=$(mktemp -d)
python3 "$root/render_redis_acl.py" "$work_dir/users.acl" --key-prefix "${TWINKLE_REDIS_KEY_PREFIX:-twinkle:h20:}"
sed '/^dir /d; /^aclfile /d' "$root/redis.conf" > "$work_dir/redis.conf"
envsubst '${TWINKLE_UPSTREAM}' < "$root/nginx.conf.template" > /etc/nginx/conf.d/default.conf
nginx -t

start_service redis-server "$work_dir/redis.conf" --dir "$TWINKLE_REDIS_DATA_DIR" --aclfile "$work_dir/users.acl"
deadline=$((SECONDS + ${TWINKLE_REDIS_START_TIMEOUT_SECONDS:-120}))
while ((SECONDS < deadline)); do
    if REDISCLI_AUTH="$TWINKLE_REDIS_PASSWORD" redis-cli --user twinkle -p 6380 PING 2>/dev/null | grep -qx PONG; then
        start_service nginx -g 'daemon off;'
        wait_for_services
    fi
    kill -0 "${pids[0]}" 2>/dev/null || break
    sleep 0.25
done
printf 'Redis did not become ready on port 6380.\n' >&2
exit 1
