#!/usr/bin/env bash
set -euo pipefail
umask 077

root=/opt/twinkle-observability
source "$root/twinkle-runtime-common.sh"
load_config "${1:-/config/twinkle-cpu2.env}"

: "${OBS_DATA_DIR:?Set the persistent observability data directory}"
: "${OBS_REDIS_URL:?Set the private Redis address}"
: "${OBS_REDIS_PASSWORD:?Set the read-only Redis password}"
: "${GF_SECURITY_ADMIN_PASSWORD:?Set the Grafana admin password}"
export OBS_REDIS_USER=${OBS_REDIS_USER:-twinkle-grafana}
export GF_SECURITY_ADMIN_USER=${GF_SECURITY_ADMIN_USER:-admin}
export GF_AUTH_ANONYMOUS_ENABLED=false GF_USERS_ALLOW_SIGN_UP=false
export GF_SECURITY_DISABLE_GRAVATAR=true GF_ANALYTICS_REPORTING_ENABLED=false
export GF_ANALYTICS_CHECK_FOR_UPDATES=false GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false
export GF_PATHS_DATA="$OBS_DATA_DIR/grafana" GF_PATHS_LOGS="$OBS_DATA_DIR/grafana/log"
export GF_PATHS_PLUGINS="$root/plugins"
work_dir=$(mktemp -d)
export GF_PATHS_PROVISIONING="$work_dir/provisioning"
mkdir -p -- "$GF_PATHS_DATA" "$GF_PATHS_LOGS"
python3 "$root/production/render_runtime.py" "$work_dir" "$OBS_DATA_DIR" \
    "${OBS_USAGE_LIMITS_FILE:-$root/production/usage-limits.example.json}" \
    "${TWINKLE_REDIS_KEY_PREFIX:-twinkle:h20:}"

cd "$OBS_DATA_DIR"
start_service mimir -config.file="$work_dir/mimir.yaml"
start_service loki -config.file="$work_dir/loki.yaml"
start_service tempo -config.file="$work_dir/tempo.yaml"
start_service /run.sh
start_service otelcol-contrib --config="$work_dir/collector.yaml"
wait_for_services
