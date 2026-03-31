#!/bin/bash

# ============================================
# Twinkle Megatron 服务启动脚本
# ============================================
# 功能：启动 Ray 集群（支持多 GPU/CPU 节点）、Prometheus 监控和 Twinkle 服务器
# 用法：./run.sh [TEMP_DIR]
# 示例：./run.sh /tmp/twinkle_ray_logs
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 配置区（根据你的环境修改）
# ============================================

# --- Ray 集群配置 ---
# Head 节点（必须是第一个启动）
# 格式："GPU设备列表:GPU数量"，如 "0,1,2,3:4"
# 如果不需要 GPU，设为空字符串 ""
HEAD_NODE="0,1,2,3:4"

# GPU Worker 节点列表（可以有多个）
# 格式：每个元素为 "GPU设备列表:GPU数量"
# 示例：("4,5,6,7:4" "8,9,10,11:4") 表示两个 GPU worker 节点
# 如果没有 GPU worker，设为空数组 ()
GPU_WORKERS=("4,5,6,7:4")

# CPU Worker 数量
# 启动指定数量的纯 CPU worker 节点
CPU_WORKER_COUNT=1

# --- 网络配置 ---
RAY_PORT=6379
RAY_ADDRESS="127.0.0.1:$RAY_PORT"

# --- 路径配置 ---
DEFAULT_TEMP_DIR="/dashscope/caches/application/ray_logs"
LOG_FILE="run.log"

# --- Prometheus 监控配置 ---
PROMETHEUS_BIN="/dashscope/caches/application/monitor/prometheus-3.10.0.linux-amd64/prometheus"
PROMETHEUS_CONFIG_SUFFIX="session_latest/metrics/prometheus/prometheus.yml"

# --- Ray 日志轮转配置 ---
export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1

# ============================================
# 参数解析
# ============================================
TEMP_DIR="${1:-$DEFAULT_TEMP_DIR}"
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

# 解析节点配置 "devices:count" -> 返回 devices 和设置 _gpu_count
parse_node_config() {
    local config="$1"
    if [ -z "$config" ]; then
        _gpu_devices=""
        _gpu_count=0
        return
    fi
    _gpu_devices="${config%%:*}"
    _gpu_count="${config##*:}"
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
echo "  - 日志文件: $LOG_FILE"
echo ""

# 检查临时目录
if [ ! -d "$TEMP_DIR" ]; then
    print_info "创建临时目录: $TEMP_DIR"
    mkdir -p "$TEMP_DIR"
fi

# ============================================
# 停止已有 Ray 集群
# ============================================
print_header "清理环境"
print_info "停止已有的 Ray 集群..."
ray stop --force 2>/dev/null || true

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
nohup python server.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# 实时显示日志
tail -f "$LOG_FILE"