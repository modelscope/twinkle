"""Embedding quality validation: compress (query, cot) pairs via vLLM condenser,
extract embeddings via TransformersModel.forward_only(task='embedding'),
report cosine similarity.

Covers three domains: basic math, code logic, open-ended reasoning.

Architecture (2 GPUs):
  - GPU 0: vLLM condenser (compression)
  - GPU 1: TransformersModel (embedding, same path as training)

Launch:
    python cookbook/sample/emb_sample.py
    EMB_MODEL=./output/embedding_lora_transformers/step_16000 python cookbook/sample/emb_sample.py
"""
import os
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

logger = get_logger()

# -- Config -------------------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
EMB_MODEL_ID = os.environ.get('EMB_MODEL', 'ms://twinkle-kit/Qwen3.5-4B-QA-emb')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 1))
EMB_MAX_LENGTH = 8192

# -- Prompts (aligned with train_embedding_full_ddp.py) -----------------------
COMPRESS_SYSTEM = """\
You are a compression and summary assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

  `## Summary`               — extreme-density text the reader reads directly.
  `## More` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Summary
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## More
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Now begin.\
"""

COMPRESS_USER = (
    'Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts.\n\n'
    '## Query (ordering hint only)\n{query}\n\n'
    '## Passage\n{text}')

EMBED_QUERY_Q = (
    'What problem does this passage address, and what skill or method is needed? '
    'Compress into a retrieval-friendly need description.')
EMBED_QUERY_COT = (
    'Extract the reusable skill: trigger conditions, key steps, and expected output. '
    'Compress into a standardized procedure for retrieval.')

# =============================================================================
# Test pairs: 4 special categories to probe embedding quality
#   cat1 same-query-different-approach: same query paired with two CoTs that
#        solve it via different methods. Both q-c sims should be high; cross-CoT
#        sim reveals whether method-difference is reflected in embedding.
#   cat2 odd-domain: niche topics likely under-represented in pretraining,
#        tests generalization beyond mainstream STEM/web text.
#   cat3 reusable-basic: foundational facts/lemmas reusable across many queries,
#        check whether they get globally similar to lots of unrelated queries.
#   cat4 mutually-interfering: pairs of queries with overlapping vocabulary or
#        confusable concepts; intra-group cross-sim should remain discriminative.
# Each entry may carry an optional 'group' key for intra-group cross analysis.
# =============================================================================
TEST_PAIRS: List[Dict[str, str]] = [
    # --- cat1: same query, different approach -------------------------------
    {
        'domain': 'cat1-sum-gauss',
        'group': 'g1-sum',
        'query': '计算 1+2+3+...+100 的和，要求使用高斯求和公式',
        'cot': (
            '使用高斯求和公式 S = n(n+1)/2，n=100 → S = 100×101/2 = 5050。'
            '此公式由首末项配对推导：(1+100)+(2+99)+...+(50+51) = 50×101 = 5050。'
            '通用形式 S = n(a₁+aₙ)/2 适用于任意等差数列。'),
    },
    {
        'domain': 'cat1-sum-loop',
        'group': 'g1-sum',
        'query': '计算 1+2+3+...+100 的和，要求用 Python 循环累加',
        'cot': (
            'Python 循环累加：\n'
            'total = 0\n'
            'for i in range(1, 101):\n'
            '    total += i\n'
            'print(total)  # 5050\n'
            '时间复杂度 O(n)，空间 O(1)。也可用 sum(range(1, 101)) 一行写完。'),
    },
    {
        'domain': 'cat1-palindrome-twoptr',
        'group': 'g1-palindrome',
        'query': '如何使用双指针原地判断一个字符串是否是回文？要求 O(1) 额外空间。',
        'cot': (
            '双指针方法：l=0, r=len(s)-1。循环 while l<r: '
            'if s[l] != s[r] return False，否则 l+=1, r-=1。'
            '时间 O(n/2)，空间 O(1)，原地比较，无需额外内存。'
            '适合超长字符串或内存敏感场景。'),
    },
    {
        'domain': 'cat1-palindrome-reverse',
        'group': 'g1-palindrome',
        'query': '如何通过反转字符串后与原串比较来判断是否是回文？优先代码简洁。',
        'cot': (
            '反转比较法：return s == s[::-1]。'
            'Python 切片 s[::-1] 创建反转副本，与原串比较。'
            '时间 O(n)，空间 O(n)（额外的反转副本）。'
            '代码极简但内存翻倍，不适合超长字符串。'),
    },

    # --- cat2: odd domain ---------------------------------------------------
    {
        'domain': 'cat2-apiculture',
        'query': '蜂王衰老后，蜂群如何更替新蜂王？',
        'cot': (
            '工蜂感知到蜂王信息素（QMP/9-ODA）浓度下降时，'
            '会建造王台并喂养选定幼虫纯蜂王浆，激发其卵巢发育成新蜂王。'
            '若旧蜂王仍在，可能发生"母女同巢"短暂共存，或处女蜂王婚飞后回巢杀死老王。'
            '若同时孵化多只新蜂王，会发生"群王厮杀"直到只剩一只。'),
    },
    {
        'domain': 'cat2-tea-ceremony',
        'query': '日本煎茶道与抹茶道的核心差异是什么？',
        'cot': (
            '抹茶道（茶の湯）：将蒸青绿茶磨成细粉，茶筅点击搅打成沫，'
            '强调禅意、和敬清寂、千利休茶道流派，重仪式与精神修炼。'
            '煎茶道：用整片茶叶热水冲泡，类似中国功夫茶，'
            '强调玉露、煎茶的香气与回甘，受明清文人茶影响，重品鉴与雅集。'
            '前者粉茶后者叶茶；前者重道后者重味。'),
    },
    {
        'domain': 'cat2-trilobite-eye',
        'query': '三叶虫的复眼是如何工作的？',
        'cot': (
            '三叶虫拥有最早的矿化复眼，由方解石（CaCO₃）单晶构成晶状体。'
            '方解石具双折射性，但三叶虫晶状体沿光轴方向（c轴）排列以消除双折射。'
            'Schizochroal 复眼（每只眼有数百大透镜）能聚焦水下图像，'
            'Holochroal 复眼（小透镜紧密排列）覆盖更广视野。'
            '部分种类还演化出"挡光罩"应对水面强光。'),
    },
    {
        'domain': 'cat2-bell-tuning',
        'query': '中国编钟为什么一钟双音？两音如何产生？',
        'cot': (
            '编钟横截面非圆形，而是合瓦形（两片瓦合起来），'
            '导致正鼓部（前后中心）和侧鼓部（侧边）有两组互不耦合的振动模态。'
            '正鼓敲击激发对称模态发出基音（正鼓音），侧鼓敲击激发反对称模态发出侧鼓音，'
            '两音通常相差小三度或大三度。'
            '通过调整钟壁厚度、钟唇形状即可独立调谐两音，'
            '实现一钟双音、节省青铜与编悬空间。'),
    },

    # --- cat3: reusable basic facts -----------------------------------------
    {
        'domain': 'cat3-trig-identity',
        'query': '三角函数的基本恒等式有哪些？',
        'cot': (
            '勾股恒等式 sin²θ+cos²θ=1，由此推 1+tan²θ=sec²θ；1+cot²θ=csc²θ。'
            '加法定理 sin(α±β)=sinα cosβ±cosα sinβ；'
            'cos(α±β)=cosα cosβ∓sinα sinβ。'
            '倍角 sin2θ=2sinθcosθ；cos2θ=1-2sin²θ=2cos²θ-1。'
            '是简化三角表达式、求积分、解微分方程、信号处理的通用工具。'),
    },
    {
        'domain': 'cat3-bigO-definition',
        'query': '什么是大O记号？如何形式化定义？',
        'cot': (
            'f(n)=O(g(n)) 当且仅当 ∃c>0,n₀>0，∀n≥n₀, |f(n)|≤c·|g(n)|。描述增长上界。'
            '常见复杂度 O(1)<O(log n)<O(n)<O(n log n)<O(n²)<O(2ⁿ)<O(n!)。'
            '配合 Ω（下界）、Θ（紧确界）使用。'
            '是算法分析、数据结构选型、时空权衡决策的通用语言。'),
    },
    {
        'domain': 'cat3-hashtable-amortized',
        'query': '哈希表的平均时间复杂度为什么是 O(1)？',
        'cot': (
            '理想哈希函数将 n 个键均匀分布到 m 个桶，装载因子 α=n/m 保持常数（典型 0.5~0.75）。'
            '单次查找/插入期望访问 1+α 次桶，即 O(1)。'
            '动态扩容偶发 O(n) 拷贝摊还到 n 次操作仍为 O(1)。'
            '最坏情况（哈希冲突或恶意构造）退化为 O(n)，'
            '链地址法+红黑树或开放寻址可缓解。'),
    },
    {
        'domain': 'cat3-pigeonhole',
        'query': '什么是鸽笼原理？',
        'cot': (
            '若把 n+1 只鸽子放进 n 个鸽笼，至少有一个笼子装 ≥2 只。'
            '推广：把 n 只鸽子放进 k 个笼子，至少有一笼装 ≥⌈n/k⌉ 只。'
            '是组合证明的核心工具：哈希冲突必然性、生日悖论、'
            '数论存在性证明（如 m 个整数中必有两数差被 n 整除）等都依赖此原理。'),
    },

    # --- cat4: mutually interfering -----------------------------------------
    {
        'domain': 'cat4-binary-search',
        'group': 'g4-log-search',
        'query': '二分查找的时间复杂度为什么是 O(log n)？',
        'cot': (
            '每次比较将搜索区间减半：n → n/2 → n/4 → … → 1。'
            '需要 log₂n 次比较即可定位目标。'
            '前提：数据有序存储（数组），随机访问 O(1)。'
            '作用于静态数组，不涉及树结构。插入删除代价 O(n)。'),
    },
    {
        'domain': 'cat4-bst-search',
        'group': 'g4-log-search',
        'query': '二叉搜索树查找的平均时间复杂度为什么是 O(log n)？',
        'cot': (
            '平衡二叉搜索树（AVL、红黑树）高度 ≈ log₂n，'
            '从根到叶最多 log n 次比较。'
            '每次比较根据键值大小决定走左或右子树。'
            '不平衡时退化为链表 O(n)，故需自平衡机制。'
            '与二分查找不同，BST 支持 O(log n) 动态插入删除。'),
    },
    {
        'domain': 'cat4-deep-shallow-copy',
        'group': 'g4-copy-ref',
        'query': '深拷贝和浅拷贝有什么区别？',
        'cot': (
            '浅拷贝 copy.copy()：创建新外壳对象，但其内部嵌套对象仍是原对象的引用。'
            '修改嵌套元素会影响原对象。'
            '深拷贝 copy.deepcopy()：递归复制所有嵌套对象，完全独立。'
            '性能开销更大但内存隔离。'
            '关注的是"对象内部结构"是否被复制，针对单个对象的内存布局。'),
    },
    {
        'domain': 'cat4-pass-by-value-ref',
        'group': 'g4-copy-ref',
        'query': '函数传值和传引用有什么区别？',
        'cot': (
            '传值：函数收到实参的副本，对形参修改不影响外部。C/C++ 默认行为。'
            '传引用：函数收到实参的引用或地址，对形参修改直接影响外部。C++ \'&\' 引用、C 指针。'
            'Python 是"传对象引用"——可变对象修改可见，重新赋值不可见。'
            '关注的是"参数传递时拷不拷贝"，针对函数调用语义。'),
    },
    {
        'domain': 'cat4-process-thread',
        'group': 'g4-concurrency',
        'query': '进程和线程的区别？',
        'cot': (
            '进程：操作系统资源分配的基本单位，独立内存空间，进程间通信代价高（IPC、管道、共享内存）。'
            '线程：进程内执行流的基本单位，共享进程地址空间，切换开销小，'
            '可通过共享内存通信但需锁同步。'
            '进程崩溃不影响其他进程；线程崩溃可能拖垮整个进程。'
            '关注的是"执行实体的资源隔离粒度"。'),
    },
    {
        'domain': 'cat4-concurrency-parallelism',
        'group': 'g4-concurrency',
        'query': '并发和并行的区别？',
        'cot': (
            '并发（concurrency）：多任务在时间上交替推进，单核也可实现，'
            '强调任务的"同时存在性"——协程、事件循环、单核多任务。'
            '并行（parallelism）：多任务在物理上同时执行，必须多核或多机器，'
            '强调任务的"同时执行性"。'
            '并发是结构（任务的组织方式），并行是执行（硬件资源的利用方式）。'
            '一个并发程序可以并行也可以不并行执行。'),
    },
]

# Negative pairs are now subsumed by cat4 group analysis (intra-group cross-sim).
NEGATIVE_PAIRS: List[Dict[str, str]] = []


# =============================================================================
# Compression via vLLM
# =============================================================================

def compress_texts(sampler, texts: List[str], query_hint: str) -> List[str]:
    """Compress a list of texts using the vLLM condenser."""
    prompts = []
    for text in texts:
        user_msg = COMPRESS_USER.format(query=query_hint, text=text)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user_msg},
        ]})

    params = SamplingParams(max_tokens=8192, temperature=0.2, top_p=0.5, num_samples=1)
    responses = sampler.sample(prompts, params)

    results = []
    for resp in responses:
        seq = resp.sequences[0] if resp and resp.sequences else None
        text = ''
        if seq and seq.decoded:
            text = seq.decoded
            text = re.sub(r'<\|[^|]+\|>', '', text).rstrip()
        results.append(text)
    return results


# =============================================================================
# Embedding extraction (TransformersModel, same path as training)
# =============================================================================

def _build_features(texts: List[str], template: Template, role: str) -> List[Dict[str, Any]]:
    """Encode texts into embedding features, matching _get_first_feature in training."""
    features = []
    for text in texts:
        if not text.strip():
            continue
        if role == 'anchor':
            feat = template.encode({'messages': [
                {'role': 'user', 'content': text},
                {'role': 'assistant', 'content': 'Match the correct response here.'},
            ]})
            feat['labels'] = [1]
        else:
            feat = template.encode({'messages': [
                {'role': 'user', 'content': 'Match the correct query here.'},
                {'role': 'assistant', 'content': text},
            ]})
            feat['labels'] = [0]
        features.append(feat)
    return features


def get_embeddings(model: TransformersModel, template: Template,
                   texts: List[str], role: str = 'anchor') -> torch.Tensor:
    """Get embeddings via forward_only(task='embedding'), same code path as training."""
    features = _build_features(texts, template, role)
    if not features:
        return torch.zeros(0)
    outputs = model.forward_only(inputs=features, task='embedding', return_logits=True)
    return outputs['embeddings']


# =============================================================================
# Group analysis helper
# =============================================================================

def print_group_matrix(pairs, q_embs, c_embs, title: str):
    """Print intra-group cross-similarity (rows=query, cols=cot) for every
    'group' tag found in pairs. Highlights how well an embedding distinguishes
    near-neighbour queries (cat1 same-query-different-approach, cat4
    mutually-interfering)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        g = p.get('group')
        if g:
            groups[g].append(i)
    if not groups:
        return
    logger.info(f'\n{"=" * 80}\n{title}\n{"=" * 80}')
    for gname, idxs in groups.items():
        logger.info(f'\n[Group: {gname}] rows=query, cols=cot, * = matched pair')
        header = ' ' * 28
        for j in idxs:
            header += f' {pairs[j]["domain"][-14:]:>15}'
        logger.info(header)
        for i in idxs:
            row = f'{pairs[i]["domain"][-28:]:<28}'
            for j in idxs:
                s = F.cosine_similarity(q_embs[i:i + 1], c_embs[j:j + 1]).item()
                mark = '*' if i == j else ' '
                row += f' {s:>13.4f}{mark}'
            logger.info(row)


# =============================================================================
# Main
# =============================================================================

def main():
    NUM_GPUS = SAMPLER_GPUS + EMB_GPUS

    # 1. Initialize Twinkle with both device groups
    device_groups = [
        DeviceGroup(name='sampler',
                    ranks=list(range(SAMPLER_GPUS)),
                    device_type='GPU',
                    gpus_per_worker=SAMPLER_GPUS),
        DeviceGroup(name='emb_model',
                    ranks=list(range(SAMPLER_GPUS, NUM_GPUS)),
                    device_type='GPU'),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # 2. vLLM condenser sampler
    sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32768,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(
        'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
        enable_thinking=False, max_length=32768)

    # 3. Embedding model (same as training: TransformersModel + InputProcessor + InfonceLoss)
    emb_model = TransformersModel(
        model_id=EMB_MODEL_ID,
        device_mesh=emb_mesh,
        remote_group='emb_model',
    )
    emb_model.set_processor(InputProcessor)
    emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    emb_template = Template(model_id=EMB_MODEL_ID, max_length=EMB_MAX_LENGTH, enable_thinking=False)

    logger.info(get_device_placement())

    # 4. Compress all pairs
    all_pairs = TEST_PAIRS + NEGATIVE_PAIRS
    queries = [p['query'] for p in all_pairs]
    cots = [p['cot'] for p in all_pairs]

    logger.info(f'Compressing {len(queries)} queries ...')
    compressed_queries = compress_texts(sampler, queries, EMBED_QUERY_Q)
    logger.info(f'Compressing {len(cots)} CoTs ...')
    compressed_cots = compress_texts(sampler, cots, EMBED_QUERY_COT)

    # Print compression results
    for i, pair in enumerate(all_pairs):
        qc, cc = compressed_queries[i], compressed_cots[i]
        logger.info(
            f'\n{"=" * 70}\n'
            f'[{pair["domain"]}] Query ({len(pair["query"])}→{len(qc)} chars):\n'
            f'  Raw: {pair["query"][:80]}...\n'
            f'  Compressed: {qc[:120]}...\n'
            f'CoT ({len(pair["cot"])}→{len(cc)} chars):\n'
            f'  Compressed: {cc[:120]}...')

    # 5. Get embeddings via TransformersModel.forward_only(task='embedding')
    logger.info('Computing query embeddings ...')
    q_embs = get_embeddings(emb_model, emb_template, compressed_queries, role='anchor')
    logger.info('Computing CoT embeddings ...')
    c_embs = get_embeddings(emb_model, emb_template, compressed_cots, role='positive')

    logger.info('Computing raw query embeddings (no compression) ...')
    raw_q_embs = get_embeddings(emb_model, emb_template, queries, role='anchor')
    logger.info('Computing raw CoT embeddings (no compression) ...')
    raw_c_embs = get_embeddings(emb_model, emb_template, cots, role='positive')

    # 6. Compute similarities
    n_positive = len(TEST_PAIRS)
    n_negative = len(NEGATIVE_PAIRS)

    logger.info(f'\n{"=" * 70}')
    logger.info('RESULTS: Cosine Similarity (compressed query ↔ compressed CoT)')
    logger.info(f'{"=" * 70}')
    logger.info(f'{"Domain":<30} {"Compressed":>12} {"Raw":>12} {"Δ":>8}')
    logger.info('-' * 70)

    pos_sims_compressed, pos_sims_raw = [], []
    neg_sims_compressed, neg_sims_raw = [], []

    for i, pair in enumerate(all_pairs):
        sim_c = F.cosine_similarity(q_embs[i:i+1], c_embs[i:i+1]).item()
        sim_r = F.cosine_similarity(raw_q_embs[i:i+1], raw_c_embs[i:i+1]).item()
        delta = sim_c - sim_r
        marker = '✓' if i < n_positive else '✗'
        logger.info(f'  {marker} {pair["domain"]:<28} {sim_c:>10.4f}   {sim_r:>10.4f}   {delta:>+.4f}')

        if i < n_positive:
            pos_sims_compressed.append(sim_c)
            pos_sims_raw.append(sim_r)
        else:
            neg_sims_compressed.append(sim_c)
            neg_sims_raw.append(sim_r)

    # Summary statistics
    avg_pos_c = sum(pos_sims_compressed) / len(pos_sims_compressed)
    avg_pos_r = sum(pos_sims_raw) / len(pos_sims_raw)
    avg_neg_c = sum(neg_sims_compressed) / len(neg_sims_compressed) if neg_sims_compressed else 0
    avg_neg_r = sum(neg_sims_raw) / len(neg_sims_raw) if neg_sims_raw else 0

    logger.info(f'\n{"=" * 70}')
    logger.info('SUMMARY')
    logger.info(f'  Positive pairs (matched):   compressed={avg_pos_c:.4f}  raw={avg_pos_r:.4f}')
    logger.info(f'  Negative pairs (mismatched): compressed={avg_neg_c:.4f}  raw={avg_neg_r:.4f}')
    logger.info(f'  Margin (pos - neg):          compressed={avg_pos_c - avg_neg_c:.4f}  '
                f'raw={avg_pos_r - avg_neg_r:.4f}')
    logger.info(f'{"=" * 70}')

    # 6. Group analysis: intra-group cross-sim for cat1/cat4 pairs.
    print_group_matrix(all_pairs, q_embs, c_embs,
                       'GROUP ANALYSIS (compressed, intra-group cross-sim)')
    print_group_matrix(all_pairs, raw_q_embs, raw_c_embs,
                       'GROUP ANALYSIS (raw, intra-group cross-sim)')

    # 7. Global cross-similarity matrix across all pairs (compressed).
    # Useful for cat3 (reusable basics) to spot whether a 'general' CoT lights
    # up against unrelated queries.
    logger.info(f'\n{"=" * 80}')
    logger.info('GLOBAL cross-similarity matrix (compressed); * = matched diagonal')
    logger.info(f'{"=" * 80}')
    n_all = len(all_pairs)
    cross_sim = F.cosine_similarity(
        q_embs[:n_all].unsqueeze(1),
        c_embs[:n_all].unsqueeze(0), dim=2)
    header = ' ' * 30
    for j in range(n_all):
        header += f' {all_pairs[j]["domain"][-7:]:>8}'
    logger.info(header)
    for i in range(n_all):
        row = f'{all_pairs[i]["domain"][-30:]:<30}'
        for j in range(n_all):
            val = cross_sim[i, j].item()
            mark = '*' if i == j else ' '
            row += f' {val:>7.4f}{mark}'
        logger.info(row)

    logger.info('Done.')


if __name__ == '__main__':
    main()
