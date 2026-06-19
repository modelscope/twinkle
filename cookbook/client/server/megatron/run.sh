#!/bin/bash

# ============================================
# Twinkle Megatron 服务启动脚本
# ============================================
# 功能：启动 Ray 集群（支持多 GPU/CPU 节点）、LGTM 观测栈和 Twinkle 服务器
#
# 用法：./run.sh [选项]
#
# 选项：
#   --head NODE          Head 节点 GPU 配置，格式 "设备列表:数量" (默认: 0,1,2,3:4)
#   --gpu-workers LIST   GPU Worker 列表，分号分隔多个节点 (默认: 4,5,6,7:4)
#   --cpu-workers N      CPU Worker 数量 (默认: 1)
#   --temp-dir DIR       Ray 临时目录 (默认: /dashscope/caches/application/ray_logs)
#   --save-dir DIR       Twinkle 模型保存目录 (默认: /dashscope/caches/application/save)
#   --server-config FILE Twinkle 服务器配置文件路径 (默认: /twinkle/cookbook/client/server/megatron/server_config.yaml)
#   --help               显示帮助信息
#
# 示例：
#   ./run.sh                                    # 使用默认配置
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
# 格式："GPU设备列表:GPU数量"，如 "0,1,2,3:4"
# 如果不需要 GPU，设为空字符串 ""
# 可通过命令行参数 $1 传入

# GPU Worker 节点列表（可以有多个）
# 格式：用分号分隔的 "GPU设备列表:GPU数量"
# 示例："4,5,6,7:4" 或 "4,5,6,7:4;8,9,10,11:4"
# 可通过命令行参数 $2 传入

# CPU Worker 数量
# 可通过命令行参数 $3 传入

# --- 网络配置 ---
RAY_PORT=6379
RAY_ADDRESS="127.0.0.1:$RAY_PORT"

# --- 路径配置 ---
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

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "用法: ./run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --head NODE          Head 节点 GPU 设备列表，逗号分隔 (默认: 0,1,2,3)"
            echo "  --gpu-workers LIST   GPU Worker 列表，分号分隔多个节点 (默认: 4,5,6,7)"
            echo "  --cpu-workers N      CPU Worker 数量 (默认: 1)"
            echo "  --temp-dir DIR       Ray 临时目录"
            echo "  --save-dir DIR       Twinkle 模型保存目录 (默认: $DEFAULT_SAVE_DIR)"
            echo "  --server-config FILE Twinkle 服务器配置文件路径 (默认: $DEFAULT_SERVER_CONFIG_FILE)"
            echo "  --help, -h           显示帮助信息"
            echo ""
            echo "示例:"
            echo "  ./run.sh                                      # 默认配置"
            echo "  ./run.sh --head '0,1,2,3' --gpu-workers '4,5,6,7'"
            echo "  ./run.sh --head '0,1,2,3,4,5,6,7'             # 单机 8 卡"
            echo "  ./run.sh --gpu-workers '4,5,6,7;8,9,10,11'    # 多 GPU Worker"
            echo "  ./run.sh --cpu-workers 4 --head ''            # 纯 CPU 模式"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
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
print_header "Twinkle Megatron 服务启动脚本"

# 打印配置信息
print_info "集群配置："
echo ""

# 解析并显示 Head 节点
parse_node_config "$HEAD_NODE"
if [ -n "$_gpu_devices" ]; then
    echo "  [Head 节点]"
    echo "    - GPU 设备: $_gpu_devices"
    echo "    - GPU 数量: $_gpu_count"
else
    echo "  [Head 节点] CPU only"
fi

# 显示 GPU Worker 节点
if [ ${#GPU_WORKERS[@]} -gt 0 ]; then
    echo ""
    echo "  [GPU Worker 节点] 共 ${#GPU_WORKERS[@]} 个"
    for i in "${!GPU_WORKERS[@]}"; do
        parse_node_config "${GPU_WORKERS[$i]}"
        echo "    Worker $((i+1)): GPU=$_gpu_devices, Count=$_gpu_count"
    done
fi

# 显示 CPU Worker
if [ "$CPU_WORKER_COUNT" -gt 0 ]; then
    echo ""
    echo "  [CPU Worker 节点] $CPU_WORKER_COUNT 个"
fi

echo ""
print_info "运行参数："
echo "  - Ray 地址: $RAY_ADDRESS"
echo "  - 临时目录: $TEMP_DIR"
echo "  - 保存目录: $TWINKLE_DEFAULT_SAVE_DIR"
echo "  - 服务配置: $SERVER_CONFIG_FILE"
echo "  - 日志文件: $LOG_FILE"
echo ""

# 检查临时目录
if [ ! -d "$TEMP_DIR" ]; then
    print_info "创建临时目录: $TEMP_DIR"
    mkdir -p "$TEMP_DIR"
fi

# ============================================
# 停止已有服务（Redis / Ray / LGTM / Twinkle）
# ============================================
print_header "清理环境"

# 停止 Twinkle server.py（twinkle.server 模块）
print_info "停止已有的 Twinkle Server..."
pkill -f "twinkle.server" 2>/dev/null || true

# 停止 vLLM 进程
print_info "停止已有的 vLLM 进程..."
pkill -if "vLLM" 2>/dev/null || true

# 等待上述进程退出
sleep 2

# 若仍有残留则强制 SIGKILL
if pgrep -f "twinkle.server" > /dev/null 2>&1; then
    print_warning "Twinkle Server 未退出，强制终止..."
    pkill -9 -f "twinkle.server" 2>/dev/null || true
fi
if pgrep -if "vLLM" > /dev/null 2>&1; then
    print_warning "vLLM 进程未退出，强制终止..."
    pkill -9 -i -f "vLLM" 2>/dev/null || true
fi

print_info "停止已有的 Ray 集群..."
ray stop --force 2>/dev/null || true

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
        exit 1
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

# ============================================
# 启动 Redis
# ============================================
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

# ============================================
# 启动 Ray Head 节点
# ============================================
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

# ============================================
# 启动 GPU Worker 节点
# ============================================
for i in "${!GPU_WORKERS[@]}"; do
    parse_node_config "${GPU_WORKERS[$i]}"
    print_info "启动 GPU Worker $((i+1)) (GPU: $_gpu_devices)..."
    CUDA_VISIBLE_DEVICES="$_gpu_devices" ray start \
        --address=$RAY_ADDRESS \
        --num-gpus=$_gpu_count
    print_success "GPU Worker $((i+1)) 启动成功！"
done

# ============================================
# 启动 CPU Worker 节点
# ============================================
if [ "$CPU_WORKER_COUNT" -gt 0 ]; then
    print_info "启动 $CPU_WORKER_COUNT 个 CPU Worker..."
    for ((i=1; i<=CPU_WORKER_COUNT; i++)); do
        CUDA_VISIBLE_DEVICES="" ray start \
            --address=$RAY_ADDRESS \
            --num-gpus=0
    done
    print_success "CPU Worker 启动成功！"
fi

# ============================================
# 显示集群状态
# ============================================
echo ""
print_info "集群状态："
ray status 2>/dev/null || true

# ============================================
# 启动 LGTM 观测栈（Grafana + OTel Collector + Prometheus + Tempo + Loki）
# ============================================
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

# ============================================
# 启动 Twinkle 服务器
# ============================================
print_header "启动 Twinkle 服务器"

print_info "日志输出到: $LOG_FILE"
echo ""

# 启动服务器并实时显示日志
touch "$LOG_FILE"  # 预创建文件，避免 tail -f 在文件尚未写入时报错
nohup python -m twinkle.server launch --config "$SERVER_CONFIG_FILE" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
print_success "Twinkle Server 已启动 (PID: $SERVER_PID)"

# 实时显示日志（阻塞进程）
tail -f "$LOG_FILE"
