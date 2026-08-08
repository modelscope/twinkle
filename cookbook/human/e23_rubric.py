"""教师 judge（失败分类 + 可迁移先验）与它的磁盘缓存。

诊断是 E23 的唯一自变量：分类不出决定性类的题一律丢弃，绝不降级成 query-only，否则自变量被稀释。
与 v1 的区别是判据从「逐条 PASS/FAIL 的代码 review」换成「单一决定性类 + 报错证据 + 可迁移先验」，
理由见 e23_prompts.FAILURE_CLASSES 上方。
"""
import collections
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from twinkle import get_logger
from twinkle_agentic.utils.llm_backup import llm_backup
from twinkle_agentic.verifier import RubricVerifier
from twinkle_agentic.verifier.rubric_verifier import RubricItem, _extract_json_obj, _short_hash

from e23_prompts import (DIAG_SYSTEM, DIAG_USER, FAILURE_CLASSES, diag_query, diag_segment,
                         render_classes)

logger = get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))

RUBRIC_WORKERS = int(os.environ.get('RUBRIC_WORKERS', 16))
# **轨迹不在缓存键里** —— 任何改变裸解轨迹的开关（executor 的 thinking / executor 模型 / 任务域）
# 都必须体现在文件名上，否则会命中旧口径的诊断。
RUBRIC_CACHE_PATH = os.environ.get(
    'RUBRIC_CACHE_PATH', os.path.join(_HERE, 'rubric_cache_global_code_execnothink_v3.jsonl'))
RUBRIC_VERSION = 'rubric_code_v3_mechanism'
# 独立根因数 >= 此值就丢题：实测（E23.t1，480 组）只坏 1 处的题救回率 0.504，坏 >=3 处的只有
# 0.244 —— 一条 90 词内的警告救不回同时坏三处的尝试，训它只是稀释 batch。
MAX_INDEPENDENT_CAUSES = 3
# 判定温度。>0 换来的是标签多样性，代价是同一道题重判可能换类 —— 正好可以拿来测标签自一致率
# （没有真值时这是唯一的准确性代理）；要逐字复现的诊断就设 0。
DIAG_TEMPERATURE = float(os.environ.get('DIAG_TEMPERATURE', 0.3))
# 教师 API 配错时每题都会失败 -> 每题都丢 -> 主循环一晚上零更新还不报错。一条都没成功过就早失败。
API_FAIL_ABORT = int(os.environ.get('API_FAIL_ABORT', 20))
_CLASS_SHORT = {code: short for code, short, _ in FAILURE_CLASSES}
# 分类表进 llm_backup 的置信度键：判据表一改，学生/教师一致性统计必须从头算。
_CLASSES_KEY = _short_hash(render_classes())
# ⭐ 提示词哈希进缓存键。之前只靠 RUBRIC_VERSION 这个手写标签，改了 DIAG_SYSTEM 却忘了改标签就会
# 全量命中旧诊断、新规则一次都不执行，而且**毫无迹象**（日志里全是缓存命中，看起来一切正常）。
# 实测踩过：加完「题面没给就弃权」那条规则后重跑，24 道题全命中 20 分钟前的旧判决。哈希兜住这个。
_PROMPT_KEY = _short_hash(DIAG_SYSTEM + DIAG_USER + render_classes())
# 先验里出现引号字面量 = 抄了本题的列名/文件名/期望值。默认只统计不丢弃：先看住这个比例，确认
# 教师是否真的守住了「删掉本题这句话依然成立」，再决定要不要收紧成硬丢。
PRIOR_REJECT_LITERALS = os.environ.get('PRIOR_REJECT_LITERALS', '0') == '1'
_LITERAL_RE = re.compile(r"'[^']{2,}'|\"[^\"]{2,}\"")


def _same_class(a: str, b: str) -> bool:
    """student/teacher 是否算一致：只比决定性类。reason/prior 是自由文本，逐字比毫无意义。"""
    ca = (_extract_json_obj(a) or {}).get('class')
    return bool(ca) and ca == (_extract_json_obj(b) or {}).get('class')


class CodeRubricVerifier(RubricVerifier):
    """代码域 judge：不走父类的逐条 PASS/FAIL，改成单一决定性类 + 证据 + 先验。

    复用父类的调用层而**不动共享代码**：父类的 _parse_diagnosis 只认 index/verdict/reason/fix，
    evidence / prior / class 会被静默丢掉。

    ⭐ 必须经 @llm_backup，不能直接调 _sample_text：本臂没有学生 sampler（build_checker 不传），
    _sample_text 此时按设计恒返回 ''，教师 API 完全由这个装饰器提供。直接调的话每道题都拿到空
    响应、被判成「不可救」写进缓存，跑一遍就把缓存永久毒化。
    """

    @llm_backup(key_params=['query', 'rubric_key'], comparator=_same_class)
    def _classify_once(self, trajectory, sampling_params, query: str = None,
                       rubric_key: str = '') -> str:
        return self._sample_text(trajectory, sampling_params, self.score_lora_path)

    def classify(self, query: str, segment_text: str) -> Optional[Dict[str, Any]]:
        """返回解析出的 JSON；**None 专指「没拿到可解析响应」**（调用层故障，调用方不得缓存）。"""
        traj = {'messages': [
            {'role': 'system', 'content': DIAG_SYSTEM},
            {'role': 'user', 'content': DIAG_USER.format(
                query=query, rubric=render_classes(), segment=segment_text)}]}
        raw = self._classify_once(
            trajectory=traj,
            sampling_params=self._diagnose_sampling_params(None, temperature=DIAG_TEMPERATURE),
            query=query, rubric_key=_CLASSES_KEY)
        return _extract_json_obj(raw)


def build_checker():
    """没有教师 API 就返回 None（调用方据此报错退出）。

    fixed_rubric / gate 只为让父类构造合法：本臂从不读 detail.scalar，gate 与 is_hard 都不生效。
    """
    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('LLM_BACKUP_BASE_URL')
            or os.environ.get('OPENAI_API_KEY')):
        return None
    return CodeRubricVerifier(
        fixed_rubric=[RubricItem(f'{c}: {d}', is_hard=False) for c, _s, d in FAILURE_CLASSES],
        gate=False)


# 这两类**按定义**就是「对不上调用方要求的某个东西」（前者：调用方读的东西没被设上；后者：该抛的
# 异常没抛），required_value 为 null 在这里不是合法答案，只能是教师在走捷径。
_CITATION_REQUIRED = {'TESTCONTRACT', 'EXCEPTION'}


def _cites_task(obj: Dict[str, Any], query: str) -> bool:
    """教师声称「单测要的这个值题面里给了」时，核验它引的原文**真的**在题面里。

    ⭐ 这一关不能只靠 DIAG_SYSTEM 的软性要求，两轮实测都被绕过：
    第一轮（只加「题面没给就弃权」的指令）—— 109/222/409 这类「标题/键名只存在于隐藏单测里」的题
    照收不误，教师写一条 "tests assert exact string equality on titles" 的先验，这句话本身正确且
    可迁移，于是自己说服自己题可救；可工程师读完仍不知道那字符串是什么，8 个候选必然一起挂。
    第二轮（加 required_value / required_value_source 引用字段）—— 同样三道题又溜过去了，因为
    教师把 required_value 填成 null，声称「本题失败与对不上指定值无关」，整关就不适用了。
    所以对上面那两类**强制**要求引用：拿不出题面原文 = 题面确实没有 = 丢题。
    """
    val = str(obj.get('required_value') or '').strip()
    cls = str(obj.get('class') or '').strip().upper()
    if not val or val.lower() == 'null':
        # 只有「这类失败本来就跟指定值无关」时才放行；两个强制类填 null 一律当没引用处理。
        return cls not in _CITATION_REQUIRED
    src = str(obj.get('required_value_source') or '').strip()
    if not src or src.lower() == 'null':
        return False                     # 教师自认题面没给 -> 谁也做不出来
    return _squash(src) in _squash(query)    # 引文对不上题面 = 编的


def _validate(obj: Any, query: str) -> Optional[Dict[str, Any]]:
    """把教师输出收敛成可用诊断；不合格返回 None -> 该题丢出训练集。

    弃权（addressable=false）、分类不在表内、拿不出报错证据、先验为空、独立根因 >= 3、单测要的值
    题面里查无出处 —— 全部按「这题没法用一条不含解法的短警告救」处理。宁可丢题也不喂空洞诊断：
    每 chunk 产出约 40 道错题只需 24 道，丢得起（实测 t1 的 beyond_k 就丢了 328 道）。
    """
    if not isinstance(obj, dict) or not obj.get('addressable'):
        return None
    if not _cites_task(obj, query):
        return None
    cls = str(obj.get('class') or '').strip().upper()
    prior = str(obj.get('prior') or '').strip()
    evidence = str(obj.get('evidence') or '').strip()
    if cls not in _CLASS_SHORT or not prior or not evidence:
        return None
    try:
        n_causes = int(obj.get('independent_causes') or 1)
    except (TypeError, ValueError):
        n_causes = 1
    if n_causes >= MAX_INDEPENDENT_CAUSES:
        return None
    if PRIOR_REJECT_LITERALS and _LITERAL_RE.search(prior):
        return None
    secondary = [str(x).strip().upper() for x in (obj.get('secondary') or [])]
    # 引用字段一并落缓存（**不进 _format**，不给 skill-gen 看）：上一轮排查时缓存里没有它们，
    # 只能靠反推才确认教师是把 required_value 填了 null 溜过去的，白绕一圈。
    return {'class': cls, 'reason': str(obj.get('reason') or '').strip(), 'prior': prior,
            'evidence': evidence, 'n_causes': n_causes,
            'required_value': str(obj.get('required_value') or '').strip(),
            'required_value_source': str(obj.get('required_value_source') or '').strip(),
            'secondary': [s for s in secondary if s in _CLASS_SHORT and s != cls]}


# 单测报错里「期望值」的两种常见形态。⭐ 只用来**统计**教师该弃权时有没有弃权，绝不拦截：实测在
# 352 道题上约两成假阳 —— 命中的字面量可能只是大小写/标点与题面不同（'Performance'、'Random
# Walk'），也可能抓到的是单测**构造输入**用的键而不是要求返回的键（BCB/524 的 'bird'/'fish'）。
# 归一化比较能消掉前一类，后一类消不掉，所以这个信号只配当监控，判决交给看得见题面的教师。
_EXPECT_RE = re.compile(r"""!=\s*(['"])(.{2,60}?)\1|KeyError:\s*(['"])(.{1,40}?)\3""")


def _squash(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def _unseen_expected_literal(evidence: str, query: str) -> bool:
    """报错要求的字面量在题面里查无此物 = 这题从题面根本做不出来，教师本该弃权。"""
    hay = _squash(query)
    for m in _EXPECT_RE.finditer(evidence or ''):
        lit = m.group(2) or m.group(4)
        if lit and _squash(lit) not in hay:
            return True
    return False


def _format(d: Dict[str, Any]) -> str:
    """诊断 -> 喂给 skill-gen 的纯文本。

    ⭐ evidence 故意**不进** skill-gen：它是单测报错原文，断言 diff 里带期望值，是最强的泄漏通道。
    它留在缓存里只为两件事 —— 逼教师把分类落到客观事实上，以及事后审计。
    """
    lines = [f"DECISIVE FAILURE: {d['class']} — {_CLASS_SHORT[d['class']]}",
             f"WHAT WENT WRONG: {d['reason']}",
             f"PRIOR THAT WOULD HAVE PREVENTED IT: {d['prior']}"]
    if d['secondary']:
        lines.append('ALSO OFF (do not write about these): ' + ', '.join(d['secondary']))
    return '\n'.join(lines)


def class_metrics(diag_texts: List[str]) -> Dict[str, float]:
    """标签退化监控：一组题用到几个决定性类、最大那类占多少。

    判据表的全部价值在于分得开：一旦某一类吃掉大半（旧表的兜底项吃了 76% 的题），所有题的诊断
    就长得一样，skill-gen 失去逐题条件化，等于退回没有 rubric 的对照臂。
    """
    codes = [m.group(1) for m in
             (re.match(r'DECISIVE FAILURE:\s*([A-Z]+)', (t or '').split('\n', 1)[0])
              for t in diag_texts) if m]
    c = collections.Counter(codes)
    return {'signal/rubric_n_classes': float(len(c)),
            'signal/rubric_top_share': max(c.values()) / len(codes) if codes else 0.0}


def cache_metrics(delta: collections.Counter) -> Dict[str, float]:
    """一步之内 rubric 侧的计数，进 train_log。

    unseen_literal 是弃权规则的自查通道：教师本该判「题面没给」却收下的题数。要和 dropped 一起
    读 —— dropped 不涨而 unseen_literal 在涨，就说明 DIAG_SYSTEM 那条规则没被遵守。
    """
    return {'signal/rubric_dropped': float(delta['dropped_unaddressable'] + delta['hit_dropped']),
            'signal/rubric_dropped_not_stated': float(delta['dropped_not_stated']),
            'signal/rubric_api_fail': float(delta['api_fail']),
            'signal/rubric_unseen_literal': float(delta['unseen_literal']),
            'signal/rubric_prior_has_literal': float(delta['prior_has_literal'])}


class RubricCache:
    """append-only jsonl + 内存索引，键 = md5(RUBRIC_VERSION, 提示词哈希, data_id)，值 = 诊断 JSON。

    能跨 run 复用的**唯一依据**是「executor 冻结在 T=0，所以同一道题的裸解轨迹在所有 run 里逐字
    相同」。轨迹不在键里，见 RUBRIC_CACHE_PATH 上方的换名要求。
    存 JSON 而不是成品文本：改 _format 的排版不必重拉 API，evidence 也留得住可审计。
    """

    def __init__(self, path: str = RUBRIC_CACHE_PATH):
        self.path = path
        self._idx: Dict[str, Any] = {}
        self.stats: collections.Counter = collections.Counter()
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._idx[rec['key']] = rec['value']
                    except Exception:
                        continue
            logger.info(f'[rubric] 缓存载入 {len(self._idx)} 条：{path}')
        self._fh = open(path, 'a', encoding='utf-8')

    def _put(self, key: str, value: Any) -> None:
        self._idx[key] = value
        self._fh.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')
        self._fh.flush()

    def get_or_diagnose(self, checker, record: Dict[str, Any], roll: Dict[str, Any]) -> str:
        """返回诊断文本；教师弃权/不可救返回 '' 并**缓存该判决**，API 失败返回 '' 但**不缓存**。

        两者都让调用方丢掉这道题，但只有前者是稳定判决 —— 把一次瞬时抖动写进缓存会永久毒化它。
        """
        key = hashlib.md5(f"{RUBRIC_VERSION}\x00{_PROMPT_KEY}\x00"
                          f"{record.get('data_id', '')}".encode('utf-8')).hexdigest()
        if key in self._idx:
            cached = self._idx[key]
            if not cached:
                self.stats['hit_dropped'] += 1
                return ''
            self.stats['hit'] += 1
            return _format(cached)
        query = diag_query(record['problem'], record['reference_answer'])
        try:
            obj = checker.classify(query, diag_segment(roll))
        except Exception as exc:
            logger.warning(f'[rubric] classify error: {exc}')
            obj = None
        if obj is None:
            # 拿不到可解析响应 = 调用层故障，**绝不能**当成「教师判不可救」写进缓存：那会把一次
            # 抖动变成永久丢题。教师彻底不通时一条都不会成功，与其空转一夜不如早失败。
            self.stats['api_fail'] += 1
            if self.stats['api_fail'] >= API_FAIL_ABORT and not self.stats['ok']:
                raise RuntimeError(
                    f'[rubric] 连续 {self.stats["api_fail"]} 次拿不到教师响应且无一成功；'
                    f'检查 LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL / LLM_BACKUP_MODEL')
            return ''
        diag = _validate(obj, query)
        if diag is None:
            # 缓存空值 = 「这题教师给不出可用分类」，是稳定判决，下次直接跳过不再花 API 钱。
            if isinstance(obj, dict) and obj.get('addressable') and not _cites_task(obj, query):
                # 单独计数：这是「题面根本没给」，与「教师主动弃权」是两种不同的丢题原因。
                self.stats['dropped_not_stated'] += 1
            self._put(key, None)
            self.stats['dropped_unaddressable'] += 1
            return ''
        if _LITERAL_RE.search(diag['prior']):
            self.stats['prior_has_literal'] += 1
        if _unseen_expected_literal(diag['evidence'], query):
            # 教师收下了一道「答案只存在于隐藏单测里」的题。不改判决（机械检测精度不够），但这个
            # 计数持续偏高就说明 DIAG_SYSTEM 的弃权规则没被遵守。
            self.stats['unseen_literal'] += 1
        self.stats['ok'] += 1
        self.stats[f"class_{diag['class']}"] += 1
        self._put(key, diag)
        return _format(diag)

    def diagnose_many(self, checker, pairs: List[Tuple[Dict, Dict]]) -> List[str]:
        """并行拉诊断（纯 API 调用，不占 GPU）。pairs = [(record, roll), ...]。"""
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(RUBRIC_WORKERS, len(pairs)))) as ex:
            return list(ex.map(lambda rb: self.get_or_diagnose(checker, rb[0], rb[1]), pairs))

    def close(self):
        self._fh.close()
