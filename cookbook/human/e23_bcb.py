"""BigCodeBench 环境层：数据加载、沙箱单测判分、模型输出解析。

与训练完全解耦 —— 这里只回答「一段模型文本能不能跑过官方单测」，不认识 skill / rubric / GRPO。
"""
import ast as _ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from twinkle import get_logger
from twinkle.dataset import Dataset, DatasetMeta

logger = get_logger()
_HERE = os.path.dirname(os.path.abspath(__file__))

# ModelScope 数据集 id 或本地 parquet 路径都行：DatasetMeta 按 os.path.exists 自行分流，
# 走本地文件时 subset / split 会被忽略。
BCB_DATASET = os.environ.get('BCB_DATASET', 'ms://bigcode/bigcodebench')
BCB_SUBSET = os.environ.get('BCB_SUBSET', 'default')
BCB_SPLIT = os.environ.get('BCB_SPLIT', 'v0.1.0_hf')
TEST_WORKERS = int(os.environ.get('TEST_WORKERS', 24))   # 跑单测的线程池（每线程一个子进程）
TEST_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', 60))   # 单题单测墙钟上限（秒）

# 需要外网 / GUI / 子进程的库：沙箱里会挂或超时，判分噪声与 skill 无关 -> 整题排除。
EXCLUDE_LIBS = {'requests', 'urllib', 'http', 'smtplib', 'socket', 'ssl', 'ftplib',
                'mechanize', 'wikipedia', 'turtle', 'tkinter', 'subprocess', 'sendgrid',
                'python_http_client', 'django', 'flask', 'flask_login', 'flask_mail',
                'flask_restful', 'flask_wtf', 'wtforms', 'multiprocessing'}
LIB_ALIAS = {'cv2': 'cv2', 'PIL': 'PIL', 'bs4': 'bs4', 'yaml': 'yaml', 'dateutil': 'dateutil',
             'Crypto': 'Crypto', 'docx': 'docx', 'pytz': 'pytz', 'psutil': 'psutil',
             'texttable': 'texttable', 'wordcloud': 'wordcloud', 'skimage': 'skimage',
             'PyPDF2': 'PyPDF2', 'sklearn': 'sklearn'}

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


# ========== 输出解析 ==========
def after_think(text: str) -> str:
    i = (text or '').rfind('</think>')
    return text[i + len('</think>'):] if i >= 0 else (text or '')


def clean_text(decoded: Optional[str]) -> str:
    """只剔 <|...|> 这类特殊 token 的**字面量**。<think> 必须保留 —— E23 要把它递给 executor。"""
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


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


def extract_skill(text: str) -> str:
    """抽 <skills> 块；任何畸形（未闭合 / 只写在 think 里 / 空块）一律返回 ''。

    只在 </think> **之后**找：写在思考过程里的 <skills> 不算产出。返回 '' 时 executor 走干净
    direct（见 skill_solve_prompt），不会拿到半截内容。
    """
    answer = after_think(text)
    s = answer.lower().rfind('<skills>')
    if s < 0:
        return ''
    inner = s + len('<skills>')
    e = answer.lower().find('</skills>', inner)
    if e < 0:
        return ''
    return re.sub(r'</?(?:skills|skill|diagnose|pitfall|strategy|think)>', '',
                  answer[inner:e].strip(), flags=re.IGNORECASE).strip()


# ========== 沙箱判分 ==========
_RUNNER = """
import unittest, sys
loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(TestCases)
res = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)
print('__BCB__', res.testsRun, len(res.failures), len(res.errors))
sys.exit(0 if res.wasSuccessful() and res.testsRun > 0 else 1)
"""


def _trim_err(err: str, limit: int = 1600) -> str:
    """保留失败测试名与异常行，砍掉冗长 traceback 帧 —— 这是喂给 rubric 的客观证据。
    随机临时目录名换成 <sandbox>，否则同一个失败在两次运行里看起来不一样。"""
    err = re.sub(r'/tmp/bcb_[A-Za-z0-9_]+', '<sandbox>', err or '')
    lines = [ln for ln in err.splitlines() if ln.strip()]
    keep = [ln for ln in lines
            if ln.startswith(('FAIL:', 'ERROR:', 'AssertionError', 'Traceback'))
            or re.match(r'^\w*(Error|Exception|Warning)\b', ln.strip())
            or ', in ' in ln]
    return '\n'.join(keep or lines[-25:])[-limit:]


def run_tests(code: str, payload: Dict[str, Any], timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
    """子进程里跑「提交代码 + 官方 test + _RUNNER」。-> {'passed', 'kind', 'error'}。"""
    if not code.strip():
        return {'passed': False, 'kind': 'no_code', 'error': 'no parseable code block'}
    if payload['entry_point'] not in code:
        return {'passed': False, 'kind': 'no_entry',
                'error': f"function {payload['entry_point']} is not defined in the submitted code"}
    tmp = tempfile.mkdtemp(prefix='bcb_')
    try:
        path = os.path.join(tmp, 'run_case.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code + '\n\n' + payload['test'] + '\n' + _RUNNER)
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            p = subprocess.run([sys.executable, path], cwd=tmp, env=env, timeout=timeout,
                               capture_output=True, text=True, errors='replace')
        except subprocess.TimeoutExpired:
            return {'passed': False, 'kind': 'timeout',
                    'error': f'the tests did not finish within {timeout}s'}
        n_tests = n_fail = n_err = 0
        for line in (p.stdout or '').splitlines():
            if line.startswith('__BCB__'):
                _, a, b, c = line.split()
                n_tests, n_fail, n_err = int(a), int(b), int(c)
        if p.returncode == 0 and n_tests > 0:
            return {'passed': True, 'kind': 'pass', 'error': ''}
        kind = 'assertion' if n_fail else ('exception' if n_err else 'import_or_syntax')
        return {'passed': False, 'kind': kind,
                'error': _trim_err((p.stderr or '').replace(tmp, '<sandbox>'))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def empty_roll() -> Dict[str, Any]:
    return {'correct': False, 'stop_reason': 'empty', 'gen_tokens': 0, 'text': '', 'code': '',
            'kind': 'no_code', 'error': ''}


def judge_seqs(pairs: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """[(采样 sequence 或 None, payload)] -> rolls。所有判分都汇合到这里。

    必须批量：单测是子进程（导入 pandas/sklearn 后典型 1-3s），一个 chunk 几百次判分串行会比同
    chunk 的 GPU 时间还长一个量级。同 (task_id, code) 只跑一次 —— T=0 的 executor 经常对同一题
    产出逐字相同的代码。
    """
    rolls: List[Dict[str, Any]] = []
    keys: List[Optional[Tuple[str, str]]] = []
    jobs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for seq, payload in pairs:
        if seq is None:
            rolls.append(empty_roll())
            keys.append(None)
            continue
        text = clean_text(getattr(seq, 'decoded', '') or '')
        code = extract_code(text)
        key = (payload['task_id'], code)
        rolls.append({'correct': False, 'stop_reason': getattr(seq, 'stop_reason', None),
                      'gen_tokens': len(getattr(seq, 'tokens', None) or []),
                      'text': text, 'code': code, 'kind': None, 'error': ''})
        keys.append(key)
        jobs.setdefault(key, payload)
    if jobs:
        todo = list(jobs)
        with ThreadPoolExecutor(max_workers=max(1, min(TEST_WORKERS, len(todo)))) as ex:
            verdicts = dict(zip(todo, ex.map(lambda k: run_tests(k[1], jobs[k]), todo)))
        for roll, key in zip(rolls, keys):
            v = verdicts.get(key) if key is not None else None
            if v is not None:
                roll['correct'] = bool(v['passed'])
                roll['kind'], roll['error'] = v['kind'], v['error']
    return rolls


# ========== 数据 ==========
# reference_answer 的字段集：判分需要的一切（code 域不是数值答案）。
_PAYLOAD_KEYS = ('task_id', 'entry_point', 'test', 'code_prompt', 'doc_struct',
                 'canonical_solution')


def _importable(lib: str) -> bool:
    try:
        return importlib.util.find_spec(LIB_ALIAS.get(lib, lib).split('.')[0]) is not None
    except Exception:
        return False


def _row_libs(row: Dict[str, Any]) -> List[str]:
    v = row.get('libs')
    if isinstance(v, str):        # 数据集里存的是 list 的字符串形式
        try:
            return list(_ast.literal_eval(v))
        except Exception:
            return []
    return list(v or [])


def _to_record(batch: Dict[str, List]) -> Dict[str, List]:
    """原始 BCB 列 -> {'data_id', 'problem', 'reference_answer'}。

    Dataset.map 强制 batched=True，所以这里收发的都是列式 batch。
    """
    return {'data_id': list(batch['task_id']),
            'problem': list(batch['instruct_prompt']),
            'reference_answer': [{k: batch[k][i] for k in _PAYLOAD_KEYS}
                                 for i in range(len(batch['task_id']))]}


def _broken_tasks(ds: Dataset, output_dir: str) -> set:
    """参考解答跑不过自己的单测 = 沙箱/依赖不可判定，不是模型的错（实测约 7.5%）。
    自检一次后落盘缓存，题数不变则复用。必须在 map 之前调用（要读原始列）。"""
    path = os.path.join(output_dir, 'bcb_broken_tasks.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                c = json.load(f)
            if int(c.get('n_tasks', -1)) == len(ds):
                return set(c['broken'])
        except Exception as exc:
            logger.warning(f'[data] 读取 {path} 失败（{exc}），重跑自检')
    logger.info(f'[data] 沙箱自检：{len(ds)} 道题跑参考解答（一次性，之后走缓存）…')
    rows = [ds[i] for i in range(len(ds))]
    jobs = [(r['code_prompt'] + (r['canonical_solution'] or ''), {k: r[k] for k in _PAYLOAD_KEYS})
            for r in rows]
    with ThreadPoolExecutor(max_workers=max(1, min(TEST_WORKERS, len(jobs)))) as ex:
        vers = list(ex.map(lambda p: run_tests(p[0], p[1]), jobs))
    broken = {r['task_id'] for r, v in zip(rows, vers) if not v['passed']}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'n_tasks': len(ds), 'broken': sorted(broken)}, f, indent=1)
    return broken


def load_records(seed: int, eval_size: int,
                 output_dir: str) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """-> (train_dataset, eval_records)，每条记录是 {'data_id', 'problem', 'reference_answer'}。

    训练侧返回 Dataset 交给调用方喂 DataLoader；holdout 是固定的一小批、每轮 eval 整体遍历，
    没有分批的意义，直接物化成 list。

    BigCodeBench 没有 difficulty 字段，所以不做分层：过滤完按 seed 洗牌，前 eval_size 道作
    holdout。题池很小（1140 -> 剔除后约 900），重复抽到的题 rubric 全部缓存命中，成本只在
    GPU rollout。
    """
    ds = Dataset(DatasetMeta(BCB_DATASET, subset_name=BCB_SUBSET, split=BCB_SPLIT))
    n_raw = len(ds)
    ds.filter(lambda r: not (set(_row_libs(r)) & EXCLUDE_LIBS))
    n_kept_libs = len(ds)
    ds.filter(lambda r: all(_importable(x) for x in _row_libs(r)))
    logger.info(f'[data] BigCodeBench: 全集 {n_raw}，剔除需外网/GUI/子进程 {n_raw - n_kept_libs}、'
                f'依赖缺失 {n_kept_libs - len(ds)} -> {len(ds)}')

    broken = _broken_tasks(ds, output_dir)
    if broken:
        ds.filter(lambda r: r['task_id'] not in broken)
        logger.info(f'[data] 剔除参考解答自己跑不过单测的题 {len(broken)} 道 -> 可用 {len(ds)}')
    ds.map(_to_record, remove_columns=ds.dataset.column_names)

    shuffled = ds.dataset.shuffle(seed=seed)
    n_eval = min(eval_size, len(shuffled)) if eval_size > 0 else 0
    eval_records = list(shuffled.select(range(n_eval)))
    train_dataset = Dataset(DatasetMeta(data=shuffled.select(range(n_eval, len(shuffled)))))
    return train_dataset, eval_records
