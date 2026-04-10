#!/bin/bash

# ============================================
# Twinkle Megatron 服务启动脚本
# ============================================
# 功能：启动 Ray 集群（支持多 GPU/CPU 节点）、Prometheus 监控和 Twinkle 服务器
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
DEFAULT_SAVE_DIR="/dashscope/caches/application/save"
DEFAULT_SERVER_CONFIG_FILE="/twinkle/cookbook/client/server/megatron/server_config.yaml"

# --- Prometheus 监控配置 ---
PROMETHEUS_BIN="/dashscope/caches/application/monitor/prometheus-3.10.0.linux-amd64/prometheus"
PROMETHEUS_CONFIG_SUFFIX="session_latest/metrics/prometheus/prometheus.yml"

# --- Ray 日志轮转配置 ---
export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1

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

PROMETHEUS_CONFIG="${TEMP_DIR}/${PROMETHEUS_CONFIG_SUFFIX}"

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
# 停止已有 Ray 集群和 Prometheus
# ============================================
print_header "清理环境"

# 停止 Twinkle server.py（twinkle.server 模块）
print_info "停止已有的 Twinkle Server..."
pkill -f "twinkle.server" 2>/dev/null || true

# 停止 vLLM 进程
print_info "停止已有的 vLLM 进程..."
pkill -f "vllm" 2>/dev/null || true

# 等待上述进程退出
sleep 2

# 若仍有残留则强制 SIGKILL
if pgrep -f "twinkle.server" > /dev/null 2>&1; then
    print_warning "Twinkle Server 未退出，强制终止..."
    pkill -9 -f "twinkle.server" 2>/dev/null || true
fi
if pgrep -f "vllm" > /dev/null 2>&1; then
    print_warning "vLLM 进程未退出，强制终止..."
    pkill -9 -f "vllm" 2>/dev/null || true
fi

print_info "停止已有的 Ray 集群..."
ray stop --force 2>/dev/null || true

print_info "停止已有的 Prometheus..."
pkill prometheus 2>/dev/null || true

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
# 启动 Prometheus 监控（可选）
# ============================================
print_header "启动监控（可选）"

PROMETHEUS_PID=""
if [ -f "$PROMETHEUS_BIN" ]; then
    print_info "检测到 Prometheus，正在启动监控服务..."

    # 等待 Ray 生成 Prometheus 配置
    sleep 2

    if [ -f "$PROMETHEUS_CONFIG" ]; then
        nohup "$PROMETHEUS_BIN" --config.file="$PROMETHEUS_CONFIG" > prometheus.log 2>&1 &
        PROMETHEUS_PID=$!
        print_success "Prometheus 监控已启动 (PID: $PROMETHEUS_PID)"
        echo "  - 监控日志: prometheus.log"
        echo "  - 配置文件: $PROMETHEUS_CONFIG"
    else
        print_warning "Prometheus 配置文件不存在，跳过监控启动"
        echo "  - 预期路径: $PROMETHEUS_CONFIG"
    fi
else
    print_warning "未检测到 Prometheus，跳过监控启动"
    echo "  - 预期路径: $PROMETHEUS_BIN"
fi

# ============================================
# 启动 Twinkle 服务器
# ============================================
print_header "启动 Twinkle 服务器"

print_info "日志输出到: $LOG_FILE"
echo ""

# 启动服务器并实时显示日志
touch "$LOG_FILE"  # 预创建文件，避免 tail -f 在文件尚未写入时报错
nohup python -m twinkle.server --config "$SERVER_CONFIG_FILE" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
print_success "Twinkle Server 已启动 (PID: $SERVER_PID)"

# 实时显示日志
tail -f "$LOG_FILE"
