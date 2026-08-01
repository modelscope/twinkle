#!/usr/bin/env bash
# ==============================================================================
# run_ablate12.sh — sequential launcher for the 12-experiment skill ablation.
#
# Reads the run plan from skill_ablate/config.py (single source of truth for order /
# dir names / think / skill-max-tokens / optional gate), then runs each experiment via
# `python -m skill_ablate.main --exp E{n}` in RUN_ORDER.
#
# Per experiment:
#   - isolated product dir  output.ablate12/<exp_dir>/
#   - idempotent: a successful run writes <exp_dir>/DONE.json (atomic, last step); both this
#     script and skill_ablate.main skip completed experiments unless FORCE=1
#   - env snapshot          output.ablate12/<exp_dir>/env_info.txt
#   - skill-max-tokens      8192 (think) / 4096 (nothink)  [from the plan]
#   - E12 (sft, optional)   SKIPPED unless RUN_SFT=1
#   - sleep between runs to let the previous Ray/vLLM engine tear down (avoid contention)
#
# Env knobs (all optional):
#   DEEPMATH_DIR=$HERE/../../../deepmath_103k   TRAIN_N=5000   MAX_UPDATES=50   EVAL_EVERY=5
#   LR=1e-6   RUN_SFT=1   FORCE=1   ONLY="E5 E6"   SLEEP=30   SWANLAB_PROJECT=twinkle
#   MIN_LEVEL=6   CHUNK_SIZE=32   (gradient-signal fix: E1/E5 audit — level<=5 all-pass
#   dominated, 16-problem chunks leave only ~6 mixed groups per update; eval split unaffected)
#   task=code 的臂（E4/E17，见 config.py）自动改走 BigCodeBench：
#   BCB_PARQUET=... CODE_CHUNK_SIZE=48 CODE_EVAL_SIZE=200 CODE_TRAIN_N=0 TEST_WORKERS=24
# ==============================================================================
set -euo pipefail

# avoid backward-pass OOM from allocator fragmentation (E6 crash: 15GiB reserved-unallocated);
# inherited by the Ray training actors via twinkle's runtime env passthrough
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# twinkle 是 editable 安装（.pth 指向 CPFS 上的 src/）；CPFS 瞬时抖动会让 site 初始化静默
# 丢弃该路径，实验启动即死在 ModuleNotFoundError: twinkle（实测复现过一次）。PYTHONPATH 兜底。
export PYTHONPATH="$(cd "$HERE/../../.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# central env file (optional): put all knobs in one place. ENV_FILE=xxx overrides the path.
ENV_FILE="${ENV_FILE:-$HERE/ablate12.env}"
if [ -f "$ENV_FILE" ]; then
    echo "[ablate12] loading env from $ENV_FILE"
    set -a; . "$ENV_FILE"; set +a
fi

OUT_ROOT="${OUT_ROOT:-$HERE/output.ablate12}"
# interpreter with the full torch/vllm/twinkle stack; conda base shells shadow `python` with a
# numpy-less interpreter, so default to the absolute path and allow PYBIN=... to override.
PYBIN="${PYBIN:-/usr/local/bin/python3}"
# DeepMath-103K (difficulty-stratified loader in skill_ablate/data.py); replaces the old
# SEAM/aops input — see skill_quality_analysis.md 组成漂移修正.
DEEPMATH_DIR="${DEEPMATH_DIR:-$(cd "$HERE/../../.." && pwd)/deepmath_103k}"
TRAIN_N="${TRAIN_N:-5000}"
EVAL_SIZE="${EVAL_SIZE:-128}"
MAX_UPDATES="${MAX_UPDATES:-50}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LR="${LR:-1e-6}"
MIN_LEVEL="${MIN_LEVEL:-6}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
SLEEP="${SLEEP:-30}"
# SWAN_PROJ alias: swanlab>=0.8 的 pydantic Settings 会解析进程 env 里的 SWANLAB_PROJECT 并报错，
# 所以外部换项目请用 SWAN_PROJ=xxx，不要 export SWANLAB_PROJECT。
# bugfix #10：ENV_FILE 的 set -a 会把文件里的 SWANLAB_PROJECT 自动 export（正好踩中上面的坑）；
# 读完值后 unset 掉 export 属性，再以普通 shell 变量重建，保证子进程 env 里没有它。
_SWAN_PROJ_VAL="${SWAN_PROJ:-${SWANLAB_PROJECT:-twinkle}}"
unset SWANLAB_PROJECT
SWANLAB_PROJECT="$_SWAN_PROJ_VAL"
RUN_SFT="${RUN_SFT:-0}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"
# leak 一律不进 reward（项目既定要求）：留空则不传 flag，由 main.py 默认值 0 生效。
# 旧版在这里写死 1.0 且第 141 行无条件传入，会静默盖掉 Python 侧默认值。
LOGP_LEAK_PENALTY="${LOGP_LEAK_PENALTY:-}"
# E16 passrate_hinge：截断铰链惩罚强度与起点、leak gate、base_tok 筛题阈（空=用 main.py 默认）
REWARD_TRUNC_PENALTY="${REWARD_TRUNC_PENALTY:-}"
REWARD_TRUNC_LO="${REWARD_TRUNC_LO:-}"
REWARD_LEAK_GATE="${REWARD_LEAK_GATE:-}"
BASE_TOK_FLOOR="${BASE_TOK_FLOOR:-}"
# kl_beta：对初始策略的锚，是唯一能提供“恢复力”对抗漂移的旋钮（reward 只能在组内排序）。
# 空=用 main.py 默认 0.01（2026-07-29 从 0.001 上调；E1-E16 既往臂全跑在 0.001）。
# RUN_TAG 同时隔离输出目录与 swanlab 实验名，用于并列跑同一 ExpSpec 的多个变体而不互相覆盖。
KL_BETA="${KL_BETA:-}"
RUN_TAG="${RUN_TAG:-}"
# --- E17 reflexion 臂专属规模 -------------------------------------------------------------
# 本臂只在【裸 executor 做错的题】上训练与评测，与其他臂不可横比（用户 2026-07-29
# 拍板：只看本实验自身趋势），所以这些规模刻意与全局默认解耦。
# 取值全部按 E16 落盘数据标定（.tmp_analysis/e17_param_calib.py，同模型/同题池/同难度门）：
#  * K=24：E16 每次更新实际只有 23.46 组（总 1173 组）。累积证据 ∝sqrt(N)，而 E16 全程
#    在文本层只累到 1.9 sigma —— 再砍组数就没任何判别力了。K=24 x 50 = 1200 组，
#    恰好追平 E16，而成本 24x8x8=1536 rollouts/chunk vs E16 实测 1501，几乎相等。
#  * CHUNK 128：实测裸错率 0.329（全量题池 527/1600，level>=6, T=0, floor=0）。
#    ⭐ 不是 0.446 —— 那是 base_tok>5000 筛选后子集的错误率（偏难），本臂 floor=0。
#    二项精算 P(凑不满 24)：chunk 64 = 74%、96 = 3.6%、112 = 0.27%、128 = 0.012%。
#    裸解开销 128 道 greedy 相当于 with-skill 的 8%，买对齐很便宜。
#  * TRAIN_N 8000：128 x 50 = 6400 次抽取，超过默认 5000 会进第二个 epoch（重复题）。
#  * EVAL 384 + --eval-min-level=MIN_LEVEL：指标只在错题子集上有信息量。旧口径（128 道
#    全难度混合）只能给 ~33 道错题，SE 0.087，趋势根本读不出来；384 道 + 难度对齐
#    训练池给 ~126 道（SE ~0.045）。正确的题跳过全部 GPU 路径，所以总成本几乎不变。
REFLEXION_K="${REFLEXION_K:-24}"
E17_CHUNK_SIZE="${E17_CHUNK_SIZE:-128}"
E17_EVAL_SIZE="${E17_EVAL_SIZE:-384}"
E17_TRAIN_N="${E17_TRAIN_N:-8000}"
# E17 专属 kl_beta（2026-07-30 拍板 0.001，回到 E1-E16 旧值）。单独立一个变量而不改
# 全局 KL_BETA，是为了不隐式改动其他臂重跑时的取值。
E17_KL_BETA="${E17_KL_BETA:-0.001}"
# --- E18 rejection_sft 臂专属 --------------------------------------------------------------
# 攒批阈值：16（用户 2026-07-30 拍板，从 128 改小）= 一个 sft_batch_size：chunk 32 每轮
# 收 ~10 条胜者，大约隔 chunk 就 fire 一次，50 次更新 ~80 chunk 可达；128 要 ~15 chunk/次。
E18_ACCUMULATE="${E18_ACCUMULATE:-16}"
# --- task=code 专属（2026-07-31 E4/E17 换到 BigCodeBench） --------------------------------
# 规模按 bcb/bcb_eval0_probe.py + 2026-07-31 dry run 的落盘数据标定，与数学默认解耦：
#  * 裸错率 **0.5625**（dry run 实测 18/32：训练侧 10/16 + eval 侧 8/16，think=on/8192）。
#    ⚠️ 不是 probe 的 0.715 —— 那是 nothink + 4096 的读数，think 开着以后模型强不少。
#    K=24 时 P(凑不满) : chunk 48 = 15.4%、56 = 1.6%、**64 = 0.1%**。组数恒定是本臂硬要求
#    （凑不满只会打印 k_short 并用更少的组训练，趋势就被抽样噪声污染），所以取 64。
#    多花的只有裸解那 16 道 greedy（with-skill 部分由 K 固定，不随 chunk 变）。
#  * EVAL 200：题池 908（1140 剔缺库/外网GUI + 沙箱自检不过的 75），错题约 112 道，SE≈0.047。
#  * TRAIN_N=0 = 用掉剩下的全部题（708 道）。chunk64 x 50 ≈ 4.5 个 epoch 重复题（拍板接受）。
#  * TEST_WORKERS：判一条 rollout = 起一个 python 子进程跑 unittest（导入 pandas/sklearn 后
#    典型 1-3s），一个 chunk 要判几百条，串行判分比同 chunk 的 GPU 时间还长。
CODE_CHUNK_SIZE="${CODE_CHUNK_SIZE:-64}"
# ★ E4（bnpo）不受 K=24 那条约束 —— 它把 chunk 里每道题都拿来训练，chunk 直接等于每次更新的
#   组数。跟着 reflexion 用 64 会白白把 rollout 数翻倍（64x8=512/更新），而且与已跑完的数学
#   E4（全局 CHUNK_SIZE=32）不再同规模、不可比。所以 view-B 的 code 臂单独用 32。
CODE_BNPO_CHUNK_SIZE="${CODE_BNPO_CHUNK_SIZE:-32}"
CODE_EVAL_SIZE="${CODE_EVAL_SIZE:-200}"
CODE_TRAIN_N="${CODE_TRAIN_N:-0}"
BCB_PARQUET="${BCB_PARQUET:-$(cd "$HERE/../../.." && pwd)/bigcodebench/bcb.parquet}"
TEST_WORKERS="${TEST_WORKERS:-24}"
TEST_TIMEOUT="${TEST_TIMEOUT:-60}"
# --- executor nothink 对照（E19 math / E20 code，2026-07-31） -------------------------------
# chunk / eval 都**不覆盖**：直接沿用同域 think 臂的值（math 128/384、code 64/200）。
# 理由一，实测：首个 E19 run 的 c0 只从 64 道里拿到 20 道错题 —— math 关 think 的裸错率约
#   0.31（baseline acc 0.69），与 think 的 0.329 基本相同，我此前估的 0.85 完全错了。
#   p=0.31 时 chunk 64 的期望错题 20±3.7，几乎每个 chunk 都凑不满 K=24，组数恒定失效。
#   code 侧 nothink p=0.622 > think 的 0.5625，chunk 64 本来就够，也无需覆盖。
# 理由二，可比性：chunk 与 eval 都与同域 think 臂逐项相同，think/nothink 才是唯一自变量。
# 保留这个 env 只为应急调参，默认空 = 不覆盖。
EXEC_NOTHINK_CHUNK_SIZE="${EXEC_NOTHINK_CHUNK_SIZE:-}"
# eval 规模**不覆盖**：两个 nothink 臂各自的对照是同域的 think 臂，评测集必须逐题相同 ——
# E19 用 E17_EVAL_SIZE=384（已跑完的 E17 数学臂就是 n=384，baseline acc 0.674 / lift +0.130），
# E20 用 CODE_EVAL_SIZE=200（与 E17 code 臂一致）。曾想为省 eval 开销把 nothink 统一压到 200，
# 那会让 E19 的评测集变成 E17 的子集、lift 曲线不再可逐题配对，省的钱不值这个代价。
# --- E4/E17/E19/E20 统一口径（2026-07-31 用户拍板） ----------------------------------------
# executor 生成预算统一 15000、skill 模型统一 8192（后者由 plan 的 smt 列给出，think=on 即 8192）。
# 统一的意义：预算不再是四个臂之间的变量，think/nothink 的对比才是单变量的。
# 连带两处必须跟着改，否则静默失效：
#  1) --max-model-len：默认 16384 装不下 prompt(约 1-2k) + 15000 输出，vLLM 会截 prompt；提到 20480。
#  2) --reward-trunc-lo：长度惩罚死区按标定比例 5500/8192 缩放到预算上 = 15000*0.671 ≈ 10000。
#     不改的话死区停在 5500（占预算 37%），会把远未撞墙的正常答案也纳入惩罚区。
UNIFIED_MAX_TOKENS="${UNIFIED_MAX_TOKENS:-15000}"
UNIFIED_MAX_MODEL_LEN="${UNIFIED_MAX_MODEL_LEN:-20480}"
UNIFIED_TRUNC_LO="${UNIFIED_TRUNC_LO:-10000}"

mkdir -p "$OUT_ROOT"

# --- pull the run plan (name \t exp_dir \t think \t smt \t optional) -------------------
# bugfix #9：用 $PYBIN（实验同一解释器）而非裸 python3，避免 plan/快照与实验环境不一致
PLAN="$("$PYBIN" skill_ablate/config.py --plan)"

snapshot_env() {  # $1 = target file
    {
        echo "=== ablate12 env snapshot @ $(date -u +%FT%TZ) ==="
        echo "host: $(hostname)"
        echo "pybin: $PYBIN"
        echo "python: $("$PYBIN" -c 'import sys;print(sys.version.split()[0])')"
        echo "torch: $("$PYBIN" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo NA)"
        echo "vllm: $("$PYBIN" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo NA)"
        echo "transformers: $("$PYBIN" -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo NA)"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
        echo "nvidia-smi:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi NA)"
        echo "GPU layout: TRAIN=${TRAIN_GPUS:-2} REF=${REF_GPUS:-2} SKILL_SAMPLER=${SKILL_SAMPLER_GPUS:-2} BASE_SAMPLER=${BASE_SAMPLER_GPUS:-2}"
        echo "LLM_BACKUP set: $([ -n "${LLM_BACKUP_API_KEY:-}${LLM_BACKUP_BASE_URL:-}${OPENAI_API_KEY:-}" ] && echo yes || echo no)"
    } > "$1"
}

echo "[ablate12] run order:"; echo "$PLAN" | awk -F'\t' '{printf "  %s -> %s (think=%s smt=%s opt=%s task=%s exec_think=%s)\n",$1,$2,$3,$4,$5,$6,$7}'

while IFS=$'\t' read -r NAME EXP_DIR THINK SMT OPTIONAL TASK EXEC_THINK; do
    [ -z "$NAME" ] && continue
    TASK="${TASK:-math}"
    EXEC_THINK="${EXEC_THINK:-on}"
    if [ -n "$ONLY" ] && ! grep -qw "$NAME" <<< "$ONLY"; then
        echo "[ablate12] $NAME skipped (not in ONLY='$ONLY')"; continue
    fi
    if [ "$OPTIONAL" = "1" ] && [ "$RUN_SFT" != "1" ]; then
        echo "[ablate12] $NAME ($EXP_DIR) skipped: optional; set RUN_SFT=1 to run"; continue
    fi

    EXP_OUT="$OUT_ROOT/$EXP_DIR${RUN_TAG:+.$RUN_TAG}"
    if [ -f "$EXP_OUT/DONE.json" ] && [ "$FORCE" != "1" ]; then
        echo "[ablate12] $NAME already done ($EXP_OUT/DONE.json); FORCE=1 to rerun"; continue
    fi
    mkdir -p "$EXP_OUT"
    snapshot_env "$EXP_OUT/env_info.txt"

    echo "======================================================================"
    echo "[ablate12] START $NAME -> $EXP_OUT  (think=$THINK skill_max_tokens=$SMT"\
"${KL_BETA:+ kl_beta=$KL_BETA}${RUN_TAG:+ tag=$RUN_TAG})"
    echo "======================================================================"
    LOG="$EXP_OUT/run.log"
    FORCE_FLAG=""
    [ "$FORCE" = "1" ] && FORCE_FLAG="--force"
    E16_FLAGS=""
    [ -n "$REWARD_TRUNC_PENALTY" ] && E16_FLAGS="$E16_FLAGS --reward-trunc-penalty $REWARD_TRUNC_PENALTY"
    [ -n "$REWARD_TRUNC_LO" ] && E16_FLAGS="$E16_FLAGS --reward-trunc-lo $REWARD_TRUNC_LO"
    [ -n "$REWARD_LEAK_GATE" ] && E16_FLAGS="$E16_FLAGS --reward-leak-gate $REWARD_LEAK_GATE"
    [ -n "$BASE_TOK_FLOOR" ] && E16_FLAGS="$E16_FLAGS --base-tok-floor $BASE_TOK_FLOOR"
    [ -n "$LOGP_LEAK_PENALTY" ] && E16_FLAGS="$E16_FLAGS --logp-leak-penalty $LOGP_LEAK_PENALTY"
    [ -n "$KL_BETA" ] && E16_FLAGS="$E16_FLAGS --kl-beta $KL_BETA"
    [ -n "$RUN_TAG" ] && E16_FLAGS="$E16_FLAGS --run-tag $RUN_TAG"
    CHUNK_ARG="$CHUNK_SIZE"
    EVAL_ARG="$EVAL_SIZE"
    TRAIN_N_ARG="$TRAIN_N"
    MIN_LEVEL_ARG="$MIN_LEVEL"
    if [ "$TASK" = "code" ]; then
        # BigCodeBench：题池/裸错率/判分方式全变，规模走 CODE_* 默认（见文件头注释）。
        # --min-level 一律传 0：BCB 没有 difficulty 字段，传非 0 只会让 data_code 打一行警告。
        CHUNK_ARG="$CODE_CHUNK_SIZE"
        [ "$NAME" = "E4" ] && CHUNK_ARG="$CODE_BNPO_CHUNK_SIZE"
        EVAL_ARG="$CODE_EVAL_SIZE"
        TRAIN_N_ARG="$CODE_TRAIN_N"
        MIN_LEVEL_ARG=0
        E16_FLAGS="$E16_FLAGS --task code --bcb-parquet $BCB_PARQUET"\
" --test-workers $TEST_WORKERS --test-timeout $TEST_TIMEOUT"
        echo "[ablate12] $NAME task=code: chunk=$CHUNK_ARG eval=$EVAL_ARG n=${TRAIN_N_ARG}(0=全池)"\
" parquet=$BCB_PARQUET test_workers=$TEST_WORKERS"
    fi
    # reflexion 家族（E17 think / E19 math-nothink / E20 code-nothink）共用同一套协议开关。
    if [ "$NAME" = "E17" ] || [ "$NAME" = "E19" ] || [ "$NAME" = "E20" ]; then
        # base_tok_floor 不在这里传：ReflexionMethod.__init__ 强制置 0 并告警，且那样
        # config 指纹里落的就是真实生效值（命令行重复传参只会制造歧义）。
        # ★ code 任务下 E17_* 这组数学标定值不适用（裸错率 0.329 -> 0.5625），保持上面
        #   CODE_* 已设好的值不动。
        if [ "$TASK" != "code" ]; then
            CHUNK_ARG="$E17_CHUNK_SIZE"
            EVAL_ARG="$E17_EVAL_SIZE"
            TRAIN_N_ARG="$E17_TRAIN_N"
            E16_FLAGS="$E16_FLAGS --eval-min-level $MIN_LEVEL"
        fi
        E16_FLAGS="$E16_FLAGS --reflexion-k $REFLEXION_K"
        # eval 口径改为 SEAM 式确定性单次（2026-07-30 拍板）：R=1 + T=0。
        # 代价：失去跨 4 个 skill 平均的降噪，题级读数从 5 档（0/.25/.5/.75/1）退为 0/1，
        # 单点 SE 约为原来的 2 倍；换来的是与 SEAM val_kwargs(n=1,do_sample=False) 同口径。
        # 也因此与 E1-E16（R=4/T=0.5）的 eval 读数不同源，不可横比。
        E16_FLAGS="$E16_FLAGS --eval-rollouts 1 --eval-skill-temperature 0.0"
        # kl_beta 回 0.001（与 E1-E16 一致）；放在这里覆盖，显式传的全局 KL_BETA 优先。
        [ -z "$KL_BETA" ] && E16_FLAGS="$E16_FLAGS --kl-beta $E17_KL_BETA"
        echo "[ablate12] $NAME reflexion: chunk=$CHUNK_ARG k=$REFLEXION_K eval=$EVAL_ARG"\
" n=$TRAIN_N_ARG task=$TASK eval_min_level=$MIN_LEVEL_ARG kl_beta=${KL_BETA:-$E17_KL_BETA}"\
" eval_rollouts=1/T=0 (hard-subset protocol; NOT comparable to E1-E16)"
    fi
    if [ "$EXEC_THINK" = "off" ]; then
        # executor 关 thinking：只加一个 flag。chunk/eval 一律沿用同域 think 臂（见文件头注释：
        # 实测 math nothink 裸错率 0.31 ≈ think 的 0.329，压小 chunk 会让 K=24 凑不满）。
        [ -n "$EXEC_NOTHINK_CHUNK_SIZE" ] && CHUNK_ARG="$EXEC_NOTHINK_CHUNK_SIZE"
        E16_FLAGS="$E16_FLAGS --executor-thinking off"
        echo "[ablate12] $NAME executor=nothink: chunk=$CHUNK_ARG eval=$EVAL_ARG"\
" (探针实测 nothink 截断 0.000、裸解 0.378>0.324、rubric 增量 +0.135 vs +0.080)"
    fi
    if [ "$NAME" = "E13" ]; then
        # SEAM 论文设置复现（2026-08-01 用户拍板）。E13 本身已是 align='seam'（SEAM EXPERIENCE_PROMPT
        # + executor 嵌套 prompt + lpem 整段 sanitize 判分）+ executor nothink（config 已改）+ skill 8192，
        # executor 预算走 main.py 默认 max-tokens=8192 / max-model-len=16384 / n-skills=8，与
        # .tmp_analysis/SEAM/scripts/train_deepmath_paper.sh 的 executor 5120+8192 / K=8 同口径。
        # 这里再把三个 run 级默认拉到 SEAM run 口径（都可被同名 env 覆盖）：
        #   * MIN_LEVEL 0：SEAM 随机全池、不挑难题；我们默认 6=只挑最难档，会压平训练集 acc 曲线，
        #     正是"我们从没见过 SEAM 那种上升曲线"的主因（见 2026-08-01 SNR 归因）。
        #   * chunk 128：= SEAM train_batch_size；bnpo(view B) 无 reflexion 的 K=24 约束，放大安全。
        #   * reward-trunc-penalty 0：SEAM reward = correct×format，无长度惩罚（train_skill_v2 头注）。
        #   * eval R=1/T=0：对齐 SEAM val_kwargs(n=1,do_sample=False)，与 E17/E19/E20 同口径，
        #     与 E1-E16(R=4/T=0.5) 不可横比。
        CHUNK_ARG="${E13_CHUNK_SIZE:-128}"
        MIN_LEVEL_ARG="${E13_MIN_LEVEL:-0}"
        [ -z "$REWARD_TRUNC_PENALTY" ] && E16_FLAGS="$E16_FLAGS --reward-trunc-penalty 0"
        E16_FLAGS="$E16_FLAGS --eval-rollouts 1 --eval-skill-temperature 0.0"
        # ---- 与 SEAM 逐行对齐的优化器参数（2026-08-01）----------------------------------
        # 上一版 E13 只对了 chunk/min_level/惩罚/eval，三个真正控制更新幅度的参数全部跑默认值，
        # 与 SEAM 差得很远（实测后果：think 25 步从 3977 塔到 1942 token、reward 0.816→0.734）。
        # SEAM 侧真值来自 scripts/train_deepmath_paper.sh + verl 的 fsdp_workers.py:198-199 归一化：
        #   ppo_mini_batch_size=20 × rollout.n=8 = 160 全局序列（再 /4gpu = 40/gpu）
        #   → 每 batch 的 optimizer step 数 = 256/40 = 7（verl data.split 的余数也算一步），
        #     twinkle 端 range(0,1024,160) 同样 7 步；且 mini<n 使 multi_step=True、启用 PPO ratio，
        #     与 verl 用 rollout 时 old_log_prob 的行为一致（旧值 mini=n → 1 步/batch、ratio 恒=1）。
        #   kl_loss_coef=0.001（main.py 默认 0.01，旧 E13 就是 0.01，整整差 10 倍）。
        # 显式同名 env 仍可覆盖。
        E16_FLAGS="$E16_FLAGS --ppo-mini-batch-size ${E13_PPO_MINI:-160}"
        [ -z "$KL_BETA" ] && E16_FLAGS="$E16_FLAGS --kl-beta ${E13_KL_BETA:-0.001}"
        # 显存：E13 8×140G 实测 micro=8（trainer.py 的 8192 自动档）OOM（132G 已用 + 4.77G），
        # micro=4（2 序列/卡）已实测跑得通，所以继续用 2×dp。
        # 注：verl 的聚合单元是 5 序列/卡（ppo_micro_batch_size_per_gpu=5），这里是 2；在
        # token_mean_scope='micro' 下这只是聚合粒度差（实测偏离真 token-mean 的倍率
        # 1.415@2 vs 1.35@5，约 5%），而且方向是更靠近 sequence-mean、长度耦合更弱，
        # 不会重新引入 'global' 那个 97 倍的“少写 token”梯度。想完全对齐需 TRAIN_FSDP=2 腾显存后设 5。
        _tmb=${E13_TRAIN_MICRO_BATCH:-$((2*${TRAIN_GPUS:-2}))}
        [ -n "${TRAIN_MICRO_BATCH:-}" ] && _tmb=$TRAIN_MICRO_BATCH
        E16_FLAGS="$E16_FLAGS --train-micro-batch $_tmb"
        echo "[ablate12] E13 SEAM-repro: chunk=$CHUNK_ARG min_level=$MIN_LEVEL_ARG train_micro_batch=$_tmb"\
" ppo_mini=${E13_PPO_MINI:-160} kl_beta=${KL_BETA:-${E13_KL_BETA:-0.001}} reward_trunc_penalty=0 eval=R1/T0"\
" executor=nothink skill_max_tokens=$SMT (executor 预算 8192/16384, K=n_skills 默认 8)"
    fi
    if [ "$NAME" = "E21" ]; then
        # 显存：E21 每卡 80G，4B 不分片 + fp32 Adam 主权重≈64G、余量仅 ~16G，micro=8（自动档）必 OOM。
        # 默认 train_micro_batch=1×dp（= TRAIN_GPUS，最小且满足 %TRAIN_DP==0）。
        # 注：token_mean_scope 已改回 'micro'（= verl 口径），所以切 micro 不再是数学等价，
        # 而是会改变聚合粒度（micro 越小越接近 sequence-mean）。E21 与其它臂横比时需记一笔。
        # 若仍 OOM：TRAIN_FSDP=2 REF_FSDP=2（分片权重、且 dp→1 允许 micro=1）。
        # 显式 TRAIN_MICRO_BATCH 优先；E21_TRAIN_MICRO_BATCH 单独可调。
        _tmb=${E21_TRAIN_MICRO_BATCH:-${TRAIN_GPUS:-2}}
        [ -n "${TRAIN_MICRO_BATCH:-}" ] && _tmb=$TRAIN_MICRO_BATCH
        E16_FLAGS="$E16_FLAGS --train-micro-batch $_tmb"
        echo "[ablate12] E21 freeform: train_micro_batch=$_tmb (80G OOM guard; micro 口径下会改变聚合粒度)"
    fi
    case "$NAME" in
        E4|E17|E19|E20)
            # 四个臂统一 executor 预算 15000 + max_model_len 20480 + 长度惩罚死区 10000，
            # 让 think/nothink 与 math/code 的对比都不夹带预算差异（2026-07-31 拍板）。
            # 显式传的 REWARD_TRUNC_LO 优先（上面已拼进 E16_FLAGS 的不会被这里覆盖）。
            E16_FLAGS="$E16_FLAGS --max-tokens $UNIFIED_MAX_TOKENS"\
" --max-model-len $UNIFIED_MAX_MODEL_LEN"
            [ -z "$REWARD_TRUNC_LO" ] && E16_FLAGS="$E16_FLAGS --reward-trunc-lo $UNIFIED_TRUNC_LO"
            echo "[ablate12] $NAME 统一口径: max_tokens=$UNIFIED_MAX_TOKENS"\
" max_model_len=$UNIFIED_MAX_MODEL_LEN skill_max_tokens=$SMT"\
" reward_trunc_lo=${REWARD_TRUNC_LO:-$UNIFIED_TRUNC_LO}"
            ;;
    esac
    if [ "$NAME" = "E18" ]; then
        # eval 是 nothink 确定性单次（trainer 侧临时切 nothink 模板；R=1 + T=0 与 E17 同拍板）。
        # 其余规模全走全局默认（用户要求"配置不变"）；攒批阈值单独可调。
        E16_FLAGS="$E16_FLAGS --e18-accumulate $E18_ACCUMULATE --eval-rollouts 1 --eval-skill-temperature 0.0"
        echo "[ablate12] E18 rejection_sft: chunk=$CHUNK_ARG eval=$EVAL_ARG n=$TRAIN_N_ARG"\
" accumulate=$E18_ACCUMULATE eval=nothink/R=1/T=0"
    fi
    set +e
    "$PYBIN" -m skill_ablate.main \
        --exp "$NAME" \
        --deepmath-dir "$DEEPMATH_DIR" \
        --n "$TRAIN_N_ARG" \
        --eval-size "$EVAL_ARG" \
        --output-dir "$EXP_OUT" \
        --skill-max-tokens "$SMT" \
        --max-updates "$MAX_UPDATES" \
        --eval-every-updates "$EVAL_EVERY" \
        --min-level "$MIN_LEVEL_ARG" \
        --chunk-size "$CHUNK_ARG" \
        --lr "$LR" \
        --swanlab-project "$SWANLAB_PROJECT" \
        $E16_FLAGS \
        $FORCE_FLAG \
        < /dev/null 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" != "0" ]; then
        echo "[ablate12] $NAME FAILED (rc=$RC); see $LOG. Stopping."; exit "$RC"
    fi
    echo "[ablate12] $NAME done. Sleeping ${SLEEP}s for engine teardown..."
    sleep "$SLEEP"
done <<< "$PLAN"

echo "[ablate12] all requested experiments finished."
