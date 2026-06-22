#!/bin/bash

# ============================================
# Twinkle Megatron 服务启动脚本
# ============================================
# 功能：启动 Ray 集群（支持多 GPU/CPU 节点）、LGTM 观测栈和 Twinkle 服务器
#
# 用法：./run.sh [选项]
#
# 选项：
#   --restart           如果已有 run.sh 实例正在运行，请求其退出并由 entrypoint 重启服务
#   --head NODE          Head 节点 GPU 设备列表，逗号分隔 (默认: 0,1,2,3)
#   --gpu-workers LIST   GPU Worker 列表，分号分隔多个节点 (默认: 4,5,6,7)
#   --cpu-workers N      CPU Worker 数量 (默认: 1)
#   --temp-dir DIR       Ray 临时目录 (默认: /dashscope/caches/application/ray_logs)
#   --save-dir DIR       Twinkle 模型保存目录 (默认: /dashscope/caches/application/save)
#   --server-config FILE Twinkle 服务器配置文件路径 (默认: /twinkle/cookbook/client/server/megatron/server_config.yaml)
#   --help               显示帮助信息
#
# 环境变量：
#   MODELSCOPE_CACHE                         默认 /dashscope/caches/application/.cache
#   TWINKLE_WORK_DIR                         默认 /dashscope/caches/application/twinkle
#   TWINKLE_RUN_EXISTING_ACTION              已有 run.sh 进程运行时的行为：exit 或 restart（默认 exit）
#   TWINKLE_RUN_RESTART_TIMEOUT_SECONDS      --restart 等待已有实例接收请求秒数（默认 120）
#
# 示例：
#   bash /twinkle/cookbook/client/server/megatron/entrypoint.sh
#                                               # 容器启动入口，负责 run.sh 保活和外部 health 检查
#   bash /twinkle/cookbook/client/server/megatron/run.sh
#                                               # 直接启动服务，前台等待 server 子进程
#   bash /twinkle/cookbook/client/server/megatron/run.sh --restart
#                                               # 更新代码后请求已有 run.sh 退出并由 entrypoint 重启
#   ./run.sh --head "0,1,2,3" --gpu-workers "4,5,6,7" --cpu-workers 1
#   ./run.sh --head "0,1,2,3" --gpu-workers "" --cpu-workers 0
#   ./run.sh --head "" --cpu-workers 4          # 纯 CPU 模式
#   ./run.sh --temp-dir /tmp/my_ray_logs        # 自定义临时目录
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 配置区（根据你的环境修改）
# ============================================

# --- Ray 集群配置 ---
# Head 节点（必须是第一个启动）
# 格式："GPU设备列表"，如 "0,1,2,3"
# 如果不需要 GPU，设为空字符串 ""
# 可通过命令行参数 $1 传入

# GPU Worker 节点列表（可以有多个）
# 格式：用分号分隔的 "GPU设备列表"
# 示例："4,5,6,7" 或 "4,5,6,7;8,9,10,11"
# 可通过命令行参数 $2 传入

# CPU Worker 数量
# 可通过命令行参数 $3 传入

# --- 网络配置 ---
RAY_PORT=6379
RAY_ADDRESS="127.0.0.1:$RAY_PORT"

# --- 路径配置 ---
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/dashscope/caches/application/.cache}"
TWINKLE_WORK_DIR="${TWINKLE_WORK_DIR:-/dashscope/caches/application/twinkle}"
DEFAULT_TEMP_DIR="/dashscope/caches/application/ray_logs"
LOG_FILE="run.log"
REDIS_LOG_FILE="/twinkle/redis.log"
DEFAULT_SAVE_DIR="/dashscope/caches/application/save"
DEFAULT_SERVER_CONFIG_FILE="/twinkle/cookbook/client/server/megatron/server_config.yaml"

# --- LGTM 版本配置（与 grafana/otel-lgtm:0.28.0 保持一致） ---
LGTM_VERSION="${LGTM_VERSION:-0.28.0}"
GRAFANA_VERSION="${GRAFANA_VERSION:-v13.0.1}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-v3.11.3}"
TEMPO_VERSION="${TEMPO_VERSION:-v2.10.5}"
LOKI_VERSION="${LOKI_VERSION:-v3.7.1}"
PYROSCOPE_VERSION="${PYROSCOPE_VERSION:-v2.0.2}"
OPENTELEMETRY_COLLECTOR_VERSION="${OPENTELEMETRY_COLLECTOR_VERSION:-v0.151.0}"
OBI_VERSION="${OBI_VERSION:-v0.9.0}"

# --- 单实例与重启请求配置 ---
RUN_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
TWINKLE_RUN_OWNER="${USER:-shared}"
TWINKLE_RUN_PID_FILE="${TWINKLE_RUN_PID_FILE:-/tmp/twinkle-megatron-run-${TWINKLE_RUN_OWNER}.pid}"
TWINKLE_RUN_RESTART_REQUEST_FILE="${TWINKLE_RUN_RESTART_REQUEST_FILE:-/tmp/twinkle-megatron-run-${TWINKLE_RUN_OWNER}.restart}"
TWINKLE_RUN_EXISTING_ACTION="${TWINKLE_RUN_EXISTING_ACTION:-exit}"
TWINKLE_RUN_RESTART_TIMEOUT_SECONDS="${TWINKLE_RUN_RESTART_TIMEOUT_SECONDS:-120}"
SERVER_PID=""
TAIL_PID=""
RESTART_REQUESTED_BY_SIGNAL=0

# ============================================
# 参数解析（支持 --key=value 或 --key value 格式）
# ============================================

# 默认值
HEAD_NODE="0,1,2,3"
GPU_WORKERS_INPUT="4,5,6,7"
CPU_WORKER_COUNT="1"
TEMP_DIR="$DEFAULT_TEMP_DIR"
SAVE_DIR="$DEFAULT_SAVE_DIR"
SERVER_CONFIG_FILE="$DEFAULT_SERVER_CONFIG_FILE"

print_usage() {
    cat <<EOF
用法: ./run.sh [选项]

选项:
  --restart           如果已有 run.sh 实例正在运行，请求其退出并由 entrypoint 重启服务
  --head NODE          Head 节点 GPU 设备列表，逗号分隔 (默认: 0,1,2,3)
  --gpu-workers LIST   GPU Worker 列表，分号分隔多个节点 (默认: 4,5,6,7)
  --cpu-workers N      CPU Worker 数量 (默认: 1)
  --temp-dir DIR       Ray 临时目录
  --save-dir DIR       Twinkle 模型保存目录 (默认: $DEFAULT_SAVE_DIR)
  --server-config FILE Twinkle 服务器配置文件路径 (默认: $DEFAULT_SERVER_CONFIG_FILE)
  --help, -h           显示帮助信息

环境变量:
  MODELSCOPE_CACHE                         默认: /dashscope/caches/application/.cache
  TWINKLE_WORK_DIR                         默认: /dashscope/caches/application/twinkle
  TWINKLE_RUN_EXISTING_ACTION              已有 run.sh 进程运行时的行为：exit 或 restart (默认: exit)
  TWINKLE_RUN_RESTART_TIMEOUT_SECONDS      --restart 等待已有实例接收请求秒数 (默认: 120)

示例:
  bash /twinkle/cookbook/client/server/megatron/entrypoint.sh
                                                # 容器启动入口，负责 run.sh 保活和外部 health 检查
  bash /twinkle/cookbook/client/server/megatron/run.sh
                                                # 直接启动服务，前台等待 server 子进程
  bash /twinkle/cookbook/client/server/megatron/run.sh --restart
                                                # 更新代码后请求已有 run.sh 退出并由 entrypoint 重启
  ./run.sh                                      # 使用默认配置
  ./run.sh --restart                            # 更新代码后请求已有 run.sh 退出并由 entrypoint 重启
  ./run.sh --head '0,1,2,3' --gpu-workers '4,5,6,7'
  ./run.sh --head '0,1,2,3,4,5,6,7'             # 单机 8 卡
  ./run.sh --gpu-workers '4,5,6,7;8,9,10,11'    # 多 GPU Worker
  ./run.sh --cpu-workers 4 --head ''            # 纯 CPU 模式
EOF
}

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --restart)
            TWINKLE_RUN_EXISTING_ACTION="restart"
            shift
            ;;
        --head)
            HEAD_NODE="$2"
            shift 2
            ;;
        --head=*)
            HEAD_NODE="${1#*=}"
            shift
            ;;
        --gpu-workers)
            GPU_WORKERS_INPUT="$2"
            shift 2
            ;;
        --gpu-workers=*)
            GPU_WORKERS_INPUT="${1#*=}"
            shift
            ;;
        --cpu-workers)
            CPU_WORKER_COUNT="$2"
            shift 2
            ;;
        --cpu-workers=*)
            CPU_WORKER_COUNT="${1#*=}"
            shift
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --temp-dir=*)
            TEMP_DIR="${1#*=}"
            shift
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --save-dir=*)
            SAVE_DIR="${1#*=}"
            shift
            ;;
        --server-config)
            SERVER_CONFIG_FILE="$2"
            shift 2
            ;;
        --server-config=*)
            SERVER_CONFIG_FILE="${1#*=}"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "\033[31m[ERROR]\033[0m 未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 将 SAVE_DIR export 给子进程（python server 通过环境变量读取）
export TWINKLE_DEFAULT_SAVE_DIR="$SAVE_DIR"

# 将分号分隔的字符串转为数组
if [ -z "$GPU_WORKERS_INPUT" ]; then
    GPU_WORKERS=()
else
    IFS=';' read -ra GPU_WORKERS <<< "$GPU_WORKERS_INPUT"
fi

# ============================================
# 辅助函数
# ============================================
print_info() {
    echo -e "\033[36m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_separator() {
    echo "============================================"
}

print_header() {
    echo ""
    print_separator
    echo -e "\033[1;34m $1 \033[0m"
    print_separator
}

is_run_script_process() {
    local pid="$1"
    local command_arg
    local command_line
    local process_cwd

    if [ -z "$pid" ] || [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    if [ -r "/proc/$pid/cmdline" ]; then
        process_cwd="$(cd "/proc/$pid/cwd" 2>/dev/null && pwd || true)"
        while IFS= read -r -d '' command_arg; do
            if is_run_script_arg "$command_arg" "$process_cwd"; then
                return 0
            fi
        done < "/proc/$pid/cmdline"
        return 1
    fi

    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    case "$command_line" in
        *"$RUN_SCRIPT_PATH"*|*" ./$(basename "$RUN_SCRIPT_PATH")"*|*" $(basename "$RUN_SCRIPT_PATH")"*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_run_script_arg() {
    local command_arg="$1"
    local process_cwd="$2"
    local command_dir

    if [ "$command_arg" = "$RUN_SCRIPT_PATH" ]; then
        return 0
    fi

    if [ "$(basename "$command_arg")" != "$(basename "$RUN_SCRIPT_PATH")" ]; then
        return 1
    fi

    case "$command_arg" in
        /*)
            [ "$command_arg" = "$RUN_SCRIPT_PATH" ]
            return
            ;;
    esac

    if [ -z "$process_cwd" ]; then
        return 1
    fi

    command_dir="$(cd "$process_cwd/$(dirname "$command_arg")" 2>/dev/null && pwd || true)"
    [ "$command_dir/$(basename "$RUN_SCRIPT_PATH")" = "$RUN_SCRIPT_PATH" ]
}

find_existing_run_pid() {
    local pid

    if [ -f "$TWINKLE_RUN_PID_FILE" ]; then
        pid="$(cat "$TWINKLE_RUN_PID_FILE" 2>/dev/null || true)"
        if is_run_script_process "$pid"; then
            echo "$pid"
            return 0
        fi
    fi

    return 1
}

register_run_instance() {
    local old_pid

    old_pid="$(find_existing_run_pid || true)"
    if [ -z "$old_pid" ]; then
        echo "$$" > "$TWINKLE_RUN_PID_FILE"
        rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
        return 0
    fi

    if [ "$TWINKLE_RUN_EXISTING_ACTION" != "restart" ]; then
        print_error "已有 run.sh 实例正在运行，退出以避免中断当前服务"
        print_error "如需主动重启，请使用 --restart 或设置 TWINKLE_RUN_EXISTING_ACTION=restart"
        exit 1
    fi

    print_warning "检测到已有 run.sh 实例 (PID: $old_pid)，准备重启..."
    echo "$$" > "$TWINKLE_RUN_RESTART_REQUEST_FILE"
    if ! kill -USR1 "$old_pid" 2>/dev/null; then
        rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
        print_error "无法向已有 run.sh 实例发送重启请求 (PID: $old_pid)"
        exit 1
    fi

    for ((i=1; i<=TWINKLE_RUN_RESTART_TIMEOUT_SECONDS; i++)); do
        if [ ! -f "$TWINKLE_RUN_RESTART_REQUEST_FILE" ]; then
            print_success "已有 run.sh 已接收重启请求，将退出并由 entrypoint 重启服务"
            exit 0
        fi
        if ! kill -0 "$old_pid" 2>/dev/null; then
            rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
            print_error "已有 run.sh 实例在处理重启请求时退出 (PID: $old_pid)"
            exit 1
        fi
        sleep 1
    done

    rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
    print_error "等待已有 run.sh 接收重启请求超时 (${TWINKLE_RUN_RESTART_TIMEOUT_SECONDS}s)"
    exit 1
}

cleanup_pid_file() {
    if [ -f "$TWINKLE_RUN_PID_FILE" ] && [ "$(cat "$TWINKLE_RUN_PID_FILE" 2>/dev/null || true)" = "$$" ]; then
        rm -f "$TWINKLE_RUN_PID_FILE"
        rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
    fi
}

require_non_negative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        print_error "$name 必须是非负整数，当前值: $value"
        exit 1
    fi
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        print_error "$name 必须是正整数，当前值: $value"
        exit 1
    fi
}

validate_runtime_config() {
    case "$TWINKLE_RUN_EXISTING_ACTION" in
        exit|restart)
            ;;
        *)
            print_error "TWINKLE_RUN_EXISTING_ACTION 只能是 exit 或 restart，当前值: $TWINKLE_RUN_EXISTING_ACTION"
            exit 1
            ;;
    esac

    require_positive_int "TWINKLE_RUN_RESTART_TIMEOUT_SECONDS" "$TWINKLE_RUN_RESTART_TIMEOUT_SECONDS"
    require_non_negative_int "CPU_WORKER_COUNT" "$CPU_WORKER_COUNT"
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" &> /dev/null; then
        print_error "缺少必需命令: $command_name"
        exit 1
    fi
}

validate_runtime_dependencies() {
    require_command ps
    require_command tail
    require_command python
    require_command ray
    require_command redis-server
    require_command redis-cli
    require_command pkill
    require_command pgrep
}

wait_for_redis_ready() {
    local timeout="${1:-30}"

    for ((i=1; i<=timeout; i++)); do
        if redis-cli -p 6380 ping 2>/dev/null | grep -q '^PONG$'; then
            return 0
        fi
        sleep 1
    done

    return 1
}

wait_for_redis_stopped() {
    local timeout="${1:-30}"

    for ((i=1; i<=timeout; i++)); do
        if command -v ss &> /dev/null; then
            if ! ss -lnt 2>/dev/null | grep -q ':6380 '; then
                return 0
            fi
        elif ! redis-cli -p 6380 ping 2>/dev/null | grep -q '^PONG$'; then
            return 0
        fi
        sleep 1
    done

    return 1
}

wait_for_lgtm_ready() {
    local lgtm_pid="$1"
    local timeout="${2:-90}"

    for ((i=1; i<=timeout; i++)); do
        if [ -f /tmp/ready ]; then
            return 0
        fi
        if ! kill -0 "$lgtm_pid" 2>/dev/null; then
            return 1
        fi
        sleep 1
    done

    return 1
}

start_log_tail() {
    tail -F "$LOG_FILE" &
    TAIL_PID=$!
    print_info "日志 tail 已启动 (PID: $TAIL_PID)"
}

stop_pid() {
    local pid="$1"
    local name="$2"

    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    print_info "停止 $name (PID: $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    print_warning "$name 未正常退出，强制终止..."
    kill -9 "$pid" 2>/dev/null || true
}

cleanup_existing_runtime() {
    stop_pid "$TAIL_PID" "日志 tail"
    stop_pid "$SERVER_PID" "Twinkle Server"
    TAIL_PID=""
    SERVER_PID=""

    print_info "停止已有的 Twinkle Server..."
    pkill -f "twinkle.server" 2>/dev/null || true

    print_info "停止已有的 vLLM 进程..."
    pkill -if "vLLM" 2>/dev/null || true

    sleep 2

    if pgrep -f "twinkle.server" > /dev/null 2>&1; then
        print_warning "Twinkle Server 未退出，强制终止..."
        pkill -9 -f "twinkle.server" 2>/dev/null || true
    fi
    if pgrep -if "vLLM" > /dev/null 2>&1; then
        print_warning "vLLM 进程未退出，强制终止..."
        pkill -9 -i -f "vLLM" 2>/dev/null || true
    fi

    print_info "停止已有的 Ray 集群..."
    ray stop --force >/dev/null 2>&1 || true

    print_info "停止已有的 Redis..."
    redis-cli -p 6380 shutdown nosave 2>/dev/null || pkill redis-server 2>/dev/null || true
    if ! wait_for_redis_stopped 30; then
        print_warning "Redis 未在 30 秒内退出，强制终止..."
        pkill -9 redis-server 2>/dev/null || true
        if ! wait_for_redis_stopped 10; then
            print_error "Redis 端口 6380 未释放"
            if command -v ss &> /dev/null; then
                ss -lntp 2>/dev/null | grep ':6380 ' || true
            fi
            return 1
        fi
    fi

    print_info "停止已有的 LGTM 观测栈..."
    pkill -f "/otel-lgtm/run-all.sh" 2>/dev/null || true
    pkill prometheus 2>/dev/null || true
    pkill grafana 2>/dev/null || true
    pkill otelcol 2>/dev/null || true
    pkill loki 2>/dev/null || true
    pkill tempo 2>/dev/null || true
    pkill pyroscope 2>/dev/null || true
}

cleanup_script_exit() {
    trap - EXIT INT TERM
    cleanup_existing_runtime || true
    cleanup_pid_file
}

handle_shutdown_signal() {
    cleanup_script_exit
    exit 143
}

handle_restart_signal() {
    RESTART_REQUESTED_BY_SIGNAL=1
}

consume_restart_request() {
    if [ "$RESTART_REQUESTED_BY_SIGNAL" -ne 1 ] && [ ! -f "$TWINKLE_RUN_RESTART_REQUEST_FILE" ]; then
        return 1
    fi

    RESTART_REQUESTED_BY_SIGNAL=0
    rm -f "$TWINKLE_RUN_RESTART_REQUEST_FILE"
    print_warning "收到 run.sh 重启请求，准备退出并由 entrypoint 重启服务"
    return 0
}

# 解析节点配置 "devices" -> 返回 devices 和自动计算 _gpu_count
# 示例: "0,1,2,3" -> devices="0,1,2,3", count=4
parse_node_config() {
    local config="$1"
    if [ -z "$config" ]; then
        _gpu_devices=""
        _gpu_count=0
        return
    fi
    _gpu_devices="$config"
    # 通过逗号数量+1计算 GPU 数量
    local comma_count=$(echo "$config" | tr -cd ',' | wc -c)
    _gpu_count=$((comma_count + 1))
}

# ============================================
# 开始启动
# ============================================
print_runtime_config() {
    print_header "Twinkle Megatron 服务启动脚本"

    print_info "集群配置："
    echo ""

    parse_node_config "$HEAD_NODE"
    if [ -n "$_gpu_devices" ]; then
        echo "  [Head 节点]"
        echo "    - GPU 设备: $_gpu_devices"
        echo "    - GPU 数量: $_gpu_count"
    else
        echo "  [Head 节点] CPU only"
    fi

    if [ ${#GPU_WORKERS[@]} -gt 0 ]; then
        echo ""
        echo "  [GPU Worker 节点] 共 ${#GPU_WORKERS[@]} 个"
        for i in "${!GPU_WORKERS[@]}"; do
            parse_node_config "${GPU_WORKERS[$i]}"
            echo "    Worker $((i+1)): GPU=$_gpu_devices, Count=$_gpu_count"
        done
    fi

    if [ "$CPU_WORKER_COUNT" -gt 0 ]; then
        echo ""
        echo "  [CPU Worker 节点] $CPU_WORKER_COUNT 个"
    fi

    echo ""
    print_info "运行参数："
    echo "  - Ray 地址: $RAY_ADDRESS"
    echo "  - 工作目录: $TWINKLE_WORK_DIR"
    echo "  - ModelScope 缓存: $MODELSCOPE_CACHE"
    echo "  - 临时目录: $TEMP_DIR"
    echo "  - 保存目录: $TWINKLE_DEFAULT_SAVE_DIR"
    echo "  - 服务配置: $SERVER_CONFIG_FILE"
    echo "  - 日志文件: $LOG_FILE"
    echo ""
}

prepare_runtime_dirs() {
    mkdir -p "$MODELSCOPE_CACHE" "$TEMP_DIR" "$SAVE_DIR"
}

start_redis() {
    print_header "启动 Redis"

    if command -v redis-server &> /dev/null && command -v redis-cli &> /dev/null; then
        print_info "启动 Redis..."
        redis-server --daemonize yes --port 6380 --save "" --appendonly no --logfile "$REDIS_LOG_FILE"
        if wait_for_redis_ready 30; then
            print_success "Redis 已启动 (port 6380)"
        else
            print_error "Redis 未能在 30 秒内启动或响应 PING (port 6380)"
            if [ -f "$REDIS_LOG_FILE" ]; then
                tail -n 50 "$REDIS_LOG_FILE"
            fi
            exit 1
        fi
    else
        print_error "未检测到 redis-server 或 redis-cli，但 server_config.yaml 使用 redis persistence"
        exit 1
    fi
}

start_ray_cluster() {
    print_header "启动 Ray 集群"

    parse_node_config "$HEAD_NODE"
    if [ -n "$_gpu_devices" ]; then
        print_info "启动 Head 节点 (GPU: $_gpu_devices)..."
        CUDA_VISIBLE_DEVICES="$_gpu_devices" ray start --head \
            --port=$RAY_PORT \
            --num-gpus=$_gpu_count \
            --disable-usage-stats \
            --include-dashboard=true \
            --temp-dir="$TEMP_DIR"
    else
        print_info "启动 Head 节点 (CPU only)..."
        CUDA_VISIBLE_DEVICES="" ray start --head \
            --port=$RAY_PORT \
            --num-gpus=0 \
            --disable-usage-stats \
            --include-dashboard=true \
            --temp-dir="$TEMP_DIR"
    fi
    print_success "Head 节点启动成功！"

    for i in "${!GPU_WORKERS[@]}"; do
        parse_node_config "${GPU_WORKERS[$i]}"
        print_info "启动 GPU Worker $((i+1)) (GPU: $_gpu_devices)..."
        CUDA_VISIBLE_DEVICES="$_gpu_devices" ray start \
            --address=$RAY_ADDRESS \
            --num-gpus=$_gpu_count
        print_success "GPU Worker $((i+1)) 启动成功！"
    done

    if [ "$CPU_WORKER_COUNT" -gt 0 ]; then
        print_info "启动 $CPU_WORKER_COUNT 个 CPU Worker..."
        for ((i=1; i<=CPU_WORKER_COUNT; i++)); do
            CUDA_VISIBLE_DEVICES="" ray start \
                --address=$RAY_ADDRESS \
                --num-gpus=0
        done
        print_success "CPU Worker 启动成功！"
    fi

    echo ""
    print_info "集群状态："
    ray status 2>/dev/null || true
}

start_lgtm() {
    print_header "启动 LGTM 观测栈（可选）"

    if [ -d "/otel-lgtm" ]; then
        if [ -f /otel-lgtm/prometheus.yaml.orig ]; then
            cp /otel-lgtm/prometheus.yaml.orig /otel-lgtm/prometheus.yaml
        fi

        print_info "启动 LGTM 观测栈..."
        rm -f /tmp/ready
        export LGTM_VERSION GRAFANA_VERSION PROMETHEUS_VERSION TEMPO_VERSION LOKI_VERSION
        export PYROSCOPE_VERSION OPENTELEMETRY_COLLECTOR_VERSION OBI_VERSION
        (cd /otel-lgtm && exec nohup ./run-all.sh > /twinkle/lgtm.log 2>&1) &
        LGTM_PID=$!

        if wait_for_lgtm_ready "$LGTM_PID" 120; then
            print_success "LGTM 观测栈已启动"
            echo "  - Grafana:   http://localhost:3000 (admin/admin)"
            echo "  - OTLP gRPC: localhost:4317"
            echo "  - OTLP HTTP: localhost:4318"
            echo "  - 日志文件:  /twinkle/lgtm.log"
        else
            print_warning "LGTM 观测栈未在 120 秒内就绪，Twinkle 将继续启动"
            echo "  - 日志文件: /twinkle/lgtm.log"
        fi
    else
        print_warning "未检测到 LGTM 观测栈 (/otel-lgtm)，跳过"
    fi
}

start_twinkle_server() {
    print_header "启动 Twinkle 服务器"

    print_info "日志输出到: $LOG_FILE"
    echo ""

    touch "$LOG_FILE"
    nohup python -m twinkle.server launch --config "$SERVER_CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    print_success "Twinkle Server 已启动 (PID: $SERVER_PID)"

    start_log_tail
}

wait_runtime() {
    print_info "Twinkle runtime 已启动，等待 server 进程..."
    while true; do
        if consume_restart_request; then
            return 0
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            print_error "Twinkle Server 进程已退出 (PID: $SERVER_PID)"
            return 1
        fi

        if ! kill -0 "$TAIL_PID" 2>/dev/null; then
            print_warning "日志 tail 进程已退出，重新启动..."
            start_log_tail
        fi

        sleep 1 || true
    done
}

run_service_once() {
    print_runtime_config
    prepare_runtime_dirs
    print_header "清理环境"
    cleanup_existing_runtime
    start_redis
    start_ray_cluster
    start_lgtm
    start_twinkle_server
    wait_runtime
}

validate_runtime_config
validate_runtime_dependencies
mkdir -p "$TWINKLE_WORK_DIR"
cd "$TWINKLE_WORK_DIR"
trap handle_restart_signal USR1
register_run_instance
trap cleanup_script_exit EXIT
trap handle_shutdown_signal INT TERM

run_service_once
exit 0
