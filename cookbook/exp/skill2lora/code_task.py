"""BigCodeBench task adapter: data / executor prompts / unit-test judging / code rubric.

为什么存在这个模块（承接 deepmath -> BFCL -> BigCodeBench 三轮 eval-0 探针的结论）：
  deepmath —— 76.8% 的失败是"没写完"，救活率几乎全由截断率解释，任何 hint（含乱码）只值一个
              wrapper；rubric 只能说"你超预算了/你在兜圈"，命中率≈随机，rubric skill 增量 −0.056。
  BFCL    —— 截断混杂消掉了，但 4B 裸解 0.861 只剩 8% headroom，且最大错误类 43% 是 ground truth
              私有口径问题，judge 看不到答案就无从判断 -> rubric skill 增量 +0.002（零）。
  BigCodeBench —— 判分是跑 unittest：对错是客观的（跑过就是对），而且**失败时机器免费给出可定位的
              证据**（异常类型 / 断言差异 / 失败用例名）。bcb/bcb_eval0_probe.py 实测（n=274，
              nothink，截断 0）：F0_none 0.378、query-only skill 0.382（+0.004）、
              rubric skill 0.513（+0.135，p=4e-5）—— 三个数据集里 rubric 第一次真正有增量。
      结论（已写入长期记忆）：rubric 有用的前提是"诊断有客观可定位的失败证据"，不是"任务是代码"。

本模块只做纯任务逻辑（加载 / prompt / 判分 / rubric 素材），**不 import train_skill_v2**，
所以 v2 可以在模块顶层 import 它而不构成循环依赖。判分口径与 bcb_eval0_probe.py 逐字同源
（extract_code / _RUNNER / run_tests / _trim_err / spec_constraints 直接搬过来），这样探针读数
与训练读数可以横比。
"""
import ast as _ast
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARQUET = os.path.join(_HERE, '..', '..', '..', 'bigcodebench', 'bcb.parquet')

# 需要外网 / GUI / 子进程的库：沙箱里会挂或超时，判分噪声与 skill 无关 -> 整题排除。
EXCLUDE_LIBS = {'requests', 'urllib', 'http', 'smtplib', 'socket', 'ssl', 'ftplib',
                'mechanize', 'wikipedia', 'turtle', 'tkinter', 'subprocess', 'sendgrid',
                'python_http_client', 'django', 'flask', 'flask_login', 'flask_mail',
                'flask_restful', 'flask_wtf', 'wtforms', 'multiprocessing'}
LIB_ALIAS = {'sklearn': 'sklearn', 'cv2': 'cv2', 'PIL': 'PIL', 'bs4': 'bs4', 'yaml': 'yaml',
             'dateutil': 'dateutil', 'Crypto': 'Crypto', 'docx': 'docx', 'pytz': 'pytz',
             'psutil': 'psutil', 'texttable': 'texttable', 'wordcloud': 'wordcloud',
             'skimage': 'skimage', 'PyPDF2': 'PyPDF2'}


# ===========================================================================
# 数据
# ===========================================================================
def _libs(rec) -> List[str]:
    v = rec.get('libs')
    if isinstance(v, str):
        try:
            return list(_ast.literal_eval(v))
        except Exception:
            return []
    return list(v or [])


def _importable(lib: str) -> bool:
    try:
        return importlib.util.find_spec(LIB_ALIAS.get(lib, lib).split('.')[0]) is not None
    except Exception:
        return False


def load_tasks(path: str, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """-> (tasks, stats)；tasks 已按 seed 洗牌，每条是一个"判分载荷"（见 payload_of）。"""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    keep, drop_missing, drop_excl = [], 0, 0
    for r in rows:
        libs = _libs(r)
        if set(libs) & EXCLUDE_LIBS:
            drop_excl += 1
            continue
        if any(not _importable(x) for x in libs):
            drop_missing += 1
            continue
        keep.append({'task_id': r['task_id'], 'instruct_prompt': r['instruct_prompt'],
                     'code_prompt': r['code_prompt'], 'test': r['test'],
                     'entry_point': r['entry_point'], 'doc_struct': r['doc_struct'],
                     'canonical_solution': r['canonical_solution'], 'libs': libs})
    random.Random(seed).shuffle(keep)
    return keep, {'raw': len(rows), 'kept': len(keep),
                  'drop_missing_lib': drop_missing, 'drop_needs_net_or_gui': drop_excl}


def payload_of(task: Dict[str, Any]) -> Dict[str, Any]:
    """训练记录里 ``reference_answer`` 的内容 —— 判分需要的一切。

    ★ 为什么塞进 reference_answer 而不是新开字段：全流水线（v2 / methods / eval_reflexion）判分
    都走 ``_parse_seq(seq, r['reference_answer'])`` 这一个入口，把载荷放这里就不用改任何签名，
    math 分支也完全不受影响。体积：test 平均 ~3KB，每题每 chunk 落盘一次，可接受。
    """
    return {k: task[k] for k in ('task_id', 'entry_point', 'test', 'code_prompt', 'doc_struct',
                                 'canonical_solution')}


# ===========================================================================
# 代码抽取 + 沙箱跑单测（与 bcb_eval0_probe.py 逐字同源）
# ===========================================================================
def after_think(text: str) -> str:
    i = (text or '').rfind('</think>')
    return text[i + len('</think>'):] if i >= 0 else (text or '')


_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def extract_code(text: str) -> str:
    """取最后一个能通过 ast.parse 的代码块；没有围栏就退化为整段（切 think 之后）。"""
    body = after_think(text or '')
    blocks = _FENCE_RE.findall(body)
    for b in reversed(blocks):
        try:
            _ast.parse(b)
            return b
        except SyntaxError:
            continue
    if blocks:
        return blocks[-1]
    try:
        _ast.parse(body)
        return body
    except SyntaxError:
        return ''


_RUNNER = """
import unittest, sys
loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(TestCases)
res = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)
print('__BCB__', res.testsRun, len(res.failures), len(res.errors))
sys.exit(0 if res.wasSuccessful() and res.testsRun > 0 else 1)
"""


def _trim_err(err: str, limit: int = 1600) -> str:
    """保留失败测试名与异常行，砍掉中间冗长的 traceback 帧（这是喂给 rubric 的客观证据）。

    随机临时目录名换成 ``<sandbox>``：traceback 帧里带着 /tmp/bcb_xxxxxxx/ 这种每次都不同的
    路径，对 judge 是纯噪声，还会让同一个失败在两次运行里看起来不一样（gen_records 里逐字
    比对失败原因时会误判成"变了"）。
    """
    err = re.sub(r'/tmp/bcb_[A-Za-z0-9_]+', '<sandbox>', err or '')
    lines = [ln for ln in err.splitlines() if ln.strip()]
    keep = [ln for ln in lines
            if ln.startswith(('FAIL:', 'ERROR:', 'AssertionError', 'Traceback'))
            or re.match(r'^\w*(Error|Exception|Warning)\b', ln.strip())
            or ', in ' in ln]
    text = '\n'.join(keep or lines[-25:])
    return text[-limit:]


def run_tests(code: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    """在独立进程 + 临时目录里跑该题自带的 unittest；返回 pass/fail + 客观报错。

    隔离手段只有"子进程 + 临时 cwd + 超时"，没有容器/seccomp：真正危险的题（外网 / GUI /
    子进程 / multiprocessing）在 load_tasks 阶段就按 EXCLUDE_LIBS 整题剔除了。
    env 里必须清掉 CUDA_VISIBLE_DEVICES —— 否则 numpy/torch 系的测试可能去抢训练用的卡。
    """
    if not code.strip():
        return {'passed': False, 'kind': 'no_code', 'error': 'no parseable code block',
                'n_tests': 0}
    if payload['entry_point'] not in code:
        return {'passed': False, 'kind': 'no_entry',
                'error': f"function {payload['entry_point']} is not defined in the submitted code",
                'n_tests': 0}
    tmp = tempfile.mkdtemp(prefix='bcb_')
    try:
        src = code + '\n\n' + payload['test'] + '\n' + _RUNNER
        path = os.path.join(tmp, 'run_case.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            p = subprocess.run([sys.executable, path], cwd=tmp, env=env, timeout=timeout,
                               capture_output=True, text=True, errors='replace')
        except subprocess.TimeoutExpired:
            return {'passed': False, 'kind': 'timeout',
                    'error': f'the tests did not finish within {timeout}s', 'n_tests': 0}
        out, err = p.stdout or '', p.stderr or ''
        n_tests = n_fail = n_err = 0
        for line in out.splitlines():
            if line.startswith('__BCB__'):
                _, a, b, c = line.split()
                n_tests, n_fail, n_err = int(a), int(b), int(c)
        if p.returncode == 0 and n_tests > 0:
            return {'passed': True, 'kind': 'pass', 'error': '', 'n_tests': n_tests}
        kind = 'assertion' if n_fail else ('exception' if n_err else 'import_or_syntax')
        return {'passed': False, 'kind': kind, 'error': _trim_err(err.replace(tmp, '<sandbox>')),
                'n_tests': n_tests}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def judge_many(items: List[Optional[Tuple[str, Any, int, Dict[str, Any]]]],
               workers: int, timeout: int) -> List[Dict[str, Any]]:
    """批量判分。``items[i]`` = (text, stop_reason, gen_tokens, payload) 或 None（无输出）。

    ★ 必须批量：单测是子进程（导入 pandas/sklearn 后典型 1-3s），而一个 E4 chunk 有 ~290 次判分。
    串行 ≈12 分钟/chunk，远超同 chunk 的 GPU 时间；线程池（--test-workers）把它压到 ~30s。
    额外去重：同一题的多个 skill 在 T=0 executor 下经常产出逐字相同的代码，去重后实测省下可观的
    子进程数（同 (task_id, code) 只跑一次）。
    """
    rolls: List[Dict[str, Any]] = []
    keys: List[Optional[Tuple[str, str]]] = []
    jobs: Dict[Tuple[str, str], Dict[str, Any]] = {}   # key -> payload（去重后的待跑集合）
    for it in items:
        if it is None:
            rolls.append(empty_roll())
            keys.append(None)
            continue
        text, stop, ntok, payload = it
        code = extract_code(text)
        key = (payload['task_id'], code)
        rolls.append({'pred': None, 'correct': False,
                      'terminated': stop != 'length', 'stop_reason': stop,
                      'gen_tokens': int(ntok or 0), 'text': text, 'code': code,
                      'kind': None, 'error': '', 'n_tests': 0})
        keys.append(key)
        jobs.setdefault(key, payload)
    if jobs:
        todo = list(jobs)
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo)))) as ex:
            res = list(ex.map(lambda k: run_tests(k[1], jobs[k], timeout), todo))
        verdicts = dict(zip(todo, res))
        for r, key in zip(rolls, keys):
            v = verdicts.get(key) if key is not None else None
            if v is None:
                continue
            r['correct'] = bool(v['passed'])
            # pred 在数学分支是"抽出来的答案，抽不到就是 None"，下游 term/answered_rate 与
            # acc/answered_pass 正是按 "pred is not None" 定义"交了可判的答案"这条通道。
            # ⚠️ 所以这里不能无条件写 kind：kind 恒非空会让 answered_rate 恒为 1.000、
            # answered_pass 退化成 candidate_pass，那两条曲线静默失效。口径对齐为：
            # 抽到代码块 -> pred = 判分结论（pass/assertion/...，便于审计）；没抽到 -> None。
            r['pred'] = v['kind'] if r['code'] else None
            r['kind'], r['error'], r['n_tests'] = v['kind'], v['error'], v['n_tests']
    return rolls


def empty_roll() -> Dict[str, Any]:
    return {'pred': None, 'correct': False, 'terminated': False, 'stop_reason': 'empty',
            'gen_tokens': 0, 'text': '', 'code': '', 'kind': 'no_code', 'error': '', 'n_tests': 0}


def selftest(tasks: List[Dict[str, Any]], workers: int, timeout: int) -> List[str]:
    """参考解答必须跑过它自己的单测 —— 跑不过说明沙箱/依赖不可判定，不是模型的错。
    返回跑不过的 task_id 列表（调用方据此剔题）。"""
    codes = [t['code_prompt'] + (t.get('canonical_solution') or '') for t in tasks]
    payloads = [payload_of(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks) or 1))) as ex:
        vers = list(ex.map(lambda p: run_tests(p[0], p[1], timeout), zip(codes, payloads)))
    return [t['task_id'] for t, v in zip(tasks, vers) if not v['passed']]


# ===========================================================================
# executor prompts
# ===========================================================================
# 本数据集的硬性交付要求（BigCodeBench 官方 instruct 模式口径）。所有臂共用。
EXEC_SYSTEM = """\
You are an expert Python engineer. You will be given a task description that ends with the exact \
import lines and function signature your solution must start with.

Deliver exactly one fenced Python code block and nothing else after it:
- Reproduce the given imports and the given function signature verbatim, including parameter \
names, order and default values.
- Add any further imports you need inside the same block; the block must run standalone.
- Return exactly the object type the task says to output. If it says the function should output \
a tuple, return a tuple in that order; if it names a matplotlib Axes, return the Axes object \
itself, not the Figure and not None.
- Implement the described behaviour for the general case, including the empty / single-element / \
missing-column edge cases and any exception the description says to raise.
- Do not call the function, do not print demonstrations, do not add tests, do not use \
`if __name__ == '__main__'`, and do not read from stdin.
- Do not include explanations outside the code block."""

# skill hint 包装语：与 v2 的数学版逐字同构（只把 "problem" 换成 "task"），保证 E4/E17 的
# executor 输入除 skill 文本外没有第二个变量。
_WRAPPER = ('Skill hint:\nFor this task, a skill-generation model has analyzed it and '
            'provided some advisory skills:\n{hint}\n'
            'Prefer using its techniques when they fit, but if you have a clearly better '
            'implementation, you may diverge. Be concise and accurate.\n')


def direct_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def skill_solve_prompt(problem: str, skill: str) -> Dict[str, Any]:
    skill = (skill or '').strip()
    if not skill:
        return direct_prompt(problem)
    return {'messages': [{'role': 'system', 'content': EXEC_SYSTEM},
                         {'role': 'user',
                          'content': problem + '\n\n' + _WRAPPER.format(hint=skill)}]}


# ===========================================================================
# skill-gen prompts
# ===========================================================================
# E4（view B, query-only）。与数学版同构：先私下把题做一遍，再只写可迁移的方法论；
# 硬禁止写出解法代码与本题字面量（否则 skill 就是抄答案，测不到"方法论有没有用"）。
SKILLGEN_SYSTEM = """\
You are a skill-generation model for a Python implementation task. Your <skills> block will be fed to a SEPARATE downstream engineer model that must write the function on its own. The engineer sees the same task description and the same required signature, but NOT your private reasoning.

First think privately: actually work out how you would implement it, including which library calls do the work. Then step back and write, inside <skills></skills>, transferable guidance for THIS TYPE of task: which library functions are the right tool and what their relevant arguments and return shapes are, how to get the return value into the exact type the task demands, which edge cases and exceptions this kind of task always has, and the common mistakes to avoid.

CRITICAL: do NOT write the solution code, and do NOT paste concrete literal values from this task. Name the API and describe the shape of the answer instead of writing it out.
Keep it to roughly one focused paragraph. Put ONLY the guidance inside <skills></skills>."""


def skillgen_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': SKILLGEN_SYSTEM},
                         {'role': 'user', 'content': f'Task:\n{problem}'}]}


# ===========================================================================
# rubric（判据 + judge prompt + 诊断素材）
# ===========================================================================
DIAG_SYSTEM = """\
You are a strategy-level code reviewer. You are given a Python task description (with the required signature), a rubric, one attempted implementation, and the REAL error that attempt produced when the task's unit tests were run. Decide PASS or FAIL for each criterion, and write the diagnosis so it becomes reusable guidance for similar tasks without seeing this attempt again.

Output STRICT JSON (no prose outside it) with this shape:
{"items": [{"index": 1, "verdict": "PASS"|"FAIL", "reason": "...", "fix": ""}], "overall": "OK"|"ISSUES", "summary": "..."}

Rules:
- Ground every FAIL in the task description or the test error you were given; do not speculate.
- The test error is authoritative evidence: if it names an exception, a wrong type or a failed assertion, the criterion it implicates must be FAIL.
- Never write out corrected code; describe the process problem at strategy level.
- A fix suggests the local correction direction without implementing it.
- Keep "reason" and "fix" concise: one short sentence each.
- Output only the JSON object."""

DIAG_USER = """\
## Task
{query}

## Rubric
{rubric}

## Attempted implementation and its test error
{segment}

Now output the diagnostic JSON object."""

# PASS = 该类问题不存在（正向陈述，与 v2._format_diagnosis / gate 语义一致）。
# 判据按 BigCodeBench 的实际失败模式组织：签名/导入、返回类型、API 用法、选库、边界、异常、逻辑。
CODE_RUBRIC = [
    ('The implementation is runnable as given: it defines the required function with the exact '
     'signature asked for and imports everything it uses', True),
    ('The value returned matches the output type and structure the task states, element for '
     'element and in the stated order', False),
    ('The library functions used exist and are called with arguments and keyword names that '
     'those functions actually accept', False),
    ('The library chosen for each step is the one the task asks for, used for its intended '
     'purpose rather than reimplemented by hand', False),
    ('Edge cases the task implies (empty input, single element, missing key or column, '
     'duplicate values) are handled instead of crashing', False),
    ('Exactly the exceptions the task specifies are raised for invalid input, and no others '
     'leak out', False),
    ('The core computation implements what the description asks, with no step skipped, '
     'inverted, or replaced by a placeholder', False),
]
# 版本号进 rubric 缓存键（rubric_cache._key）：判据一改旧诊断必须失效。code 与 math 的诊断
# 还额外分文件存（trainer 按 task 选文件名），双保险。
RUBRIC_VERSION = 'rubric_code_v1'


def spec_constraints(payload: Dict[str, Any]) -> str:
    """任务自身声明的硬约定（签名、必需库、返回规格、应抛异常、文档示例）。

    只用题面信息、不含参考解答 —— 训练时同样拿得到，所以是"可得且非泄漏"的判据依据。
    ⚠️ BFCL 那边同类做法（schema_constraints）没能提升命中率，因为那边的对错定义在 gt 私有约定
    里；这里返回类型/异常是题面明写的，所以这次它是真判据。
    """
    try:
        doc = payload['doc_struct']
        doc = json.loads(doc) if isinstance(doc, str) else (doc or {})
    except Exception:
        doc = {}
    lines = [f"- required signature (must be reproduced verbatim):\n{payload['code_prompt'].strip()}"]
    for key, label in (('reqs', 'must use these libraries'), ('returns', 'must return'),
                       ('raises', 'must raise'), ('params', 'parameters')):
        vals = [str(x).strip() for x in (doc.get(key) or []) if str(x).strip()]
        if vals:
            lines.append(f'- {label}: ' + '; '.join(vals)[:400])
    ex = [str(x) for x in (doc.get('examples') or [])][:8]
    if ex:
        lines.append('- documented example calls:\n    ' + '\n    '.join(ex))
    return '\n'.join(lines)


def diag_query(problem: str, payload: Dict[str, Any]) -> str:
    return problem + '\n\nHard requirements declared by the task:\n' + spec_constraints(payload)


def diag_segment(roll: Dict[str, Any]) -> str:
    """★ rubric 路线唯一真正有效的一处：给 judge 的不是"输出全文"，而是**提交的代码 + 单测真实
    报错**。<think> 已在 extract_code 里切掉。BFCL 那轮 judge 手上没有任何客观证据，命中率 25%
    ≈ 随机；这里报错是客观事实且不含参考解答。"""
    return (f"### Submitted code\n```python\n{roll.get('code') or '(no parseable code block)'}\n```"
            f"\n\n### Result of running the task's unit tests\n"
            f"outcome: {roll.get('kind') or 'unknown'}\n"
            f"{roll.get('error') or '(no error output)'}")


# ===========================================================================
# leak / skill 文本监控（代码域口径）
# ===========================================================================
def _canon_lines(payload: Dict[str, Any]) -> List[str]:
    out = []
    for ln in (payload.get('canonical_solution') or '').splitlines():
        s = ln.strip()
        if len(s) >= 20 and not s.startswith(('#', 'import ', 'from ', 'def ', 'return')):
            out.append(s)
    return out


def leaked(skill: str, payload: Any) -> bool:
    """代码域 leak = skill 里出现了参考解答的实质代码行（>=20 字符、非 import/def/注释）。
    与数学域一致：**只做监控，永不进 reward**（项目既定规则）。"""
    if not skill or not isinstance(payload, dict):
        return False
    return any(ln in skill for ln in _canon_lines(payload))


def skill_has_code(skill: str) -> bool:
    """skill 退化监控：本该是方法论的块里出现了代码围栏或成段 def/return。
    数学域的 digit_fraction / no_math_sentence 在代码域无意义，用这条替代。"""
    s = skill or ''
    if '```' in s:
        return True
    return bool(re.search(r'^\s*(def |return |import |for .*:|if .*:)', s, re.M))
