# -*- coding: utf-8 -*-
"""开跑前自检：把「跑 10 小时才发现环境不对」提前到 30 秒内暴露。

⭐ 为何必须存在：judge 的失败是**静默**的 —— pytest 缺失不会让进程崩，
只会让每道题都判 incorrect，日志上表现为 baseline_accuracy=0 / n_wrong=64/64，
而 collected 一直是 0。B 机曾因此白跑 10 小时 65 个 chunk。

⚠️ 关键点：judge 用 `subprocess.run([sys.executable, '_run.py'])` 起子进程，
所以 pytest 必须装在**启动脚本所用的那个解释器**里，不是 `which pytest` 指的那个。
本脚本用 sys.executable 自查，跟 judge 走同一条路径。

用法：python3 preflight.py   （用你打算启动采集的同一个 python3）
"""
import os
import subprocess
import sys
import tempfile

FAIL, WARN = [], []


def ck(name, cond, note='', hard=True):
    if not cond:
        (FAIL if hard else WARN).append(name)
    tag = 'OK ' if cond else ('BAD' if hard else 'WARN')
    print('  [%s] %-44s %s' % (tag, name, note))


print('=== 0. 解释器 ===')
print('  sys.executable = %s' % sys.executable)
print('  version        = %s' % sys.version.split()[0])

print()
print('=== 1. judge 沙箱（最关键，静默失败源）===')
# 完整复刻 judge 的执行路径：子进程 + pytest 跑一个必过的单测
# ⭐ 完整照抄 e18_kodcode.run_tests 的真实结构，而不是自己另写一个 pytest 调用：
#   - solution.py         被测代码（单测靠 `from solution import X` 取）
#   - test_solution.py    单测文件（pytest 只收集 test_*.py，名字不能改）
#   - _run.py             runner，pytest.main 显式指定 test_solution.py
#   - subprocess + cwd=tmp + sys.executable
# 只有走同一条路径，「通过」才真的等价于 judge 会判通过。
_RUNNER = '''import sys, pytest
rc = pytest.main(['-q', '--no-header', '-p', 'no:cacheprovider',
                  '--tb=short', 'test_solution.py'])
print('__KOD__', rc)
sys.exit(0 if int(rc) == 0 else 1)
'''
_SOLUTION = 'def add(a, b):\n    return a + b\n'
_TEST = 'from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n'

with tempfile.TemporaryDirectory() as td:
    for name, body in (('solution.py', _SOLUTION),
                       ('test_solution.py', _TEST),
                       ('_run.py', _RUNNER)):
        with open(os.path.join(td, name), 'w') as f:
            f.write(body)
    try:
        env = dict(os.environ, PYTHONHASHSEED='0')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        r = subprocess.run([sys.executable, '_run.py'], cwd=td, env=env,
                           capture_output=True, text=True, timeout=120)
        ok = (r.returncode == 0)
        last = (r.stdout or r.stderr).strip().split('\n')[-1][:60]
        ck('judge 沙箱可判对一份正确解', ok, 'rc=%d  %s' % (r.returncode, last))
        if not ok:
            print('      -> 修复：%s -m pip install pytest' % sys.executable)
            print('      -> 完整输出：')
            for ln in (r.stdout + r.stderr).strip().split('\n')[-6:]:
                print('         %s' % ln[:100])
    except Exception as e:
        ck('judge 沙箱可判对一份正确解', False, str(e)[:70])

print()
print('=== 2. 依赖包 ===')
import importlib.metadata as _md
# A 机实测版本，作为对照基线
BASE = {'pytest': '9.1.1', 'vllm': '0.23.0', 'torch': '2.11.0+cu130',
        'transformers': '5.14.1', 'modelscope': '1.38.1', 'datasets': '4.8.4',
        'openai': '2.45.0', 'numpy': '2.5.1'}
for pkg, want in BASE.items():
    try:
        got = _md.version(pkg)
        # 版本不一致只告警：主版本差异才真会出问题，补丁号差异通常无害
        same_major = got.split('.')[0] == want.split('.')[0]
        ck(pkg, True, '%s (A机 %s)%s' % (got, want, '' if same_major else '  <- 主版本不同'))
        if not same_major:
            WARN.append(pkg + '-major')
    except Exception:
        ck(pkg, False, '未安装 (A机 %s)' % want)

print()
print('=== 3. 教师 API（rubric 的唯一来源）===')
# 不发真请求，只查变量在不在：缺 key 会让 build_checker 返回 None,
# 后果是 rubric 全空 -> n_rubric_missing == 题数 -> 一条都收不到
for v in ('LLM_BACKUP_API_KEY', 'LLM_BACKUP_BASE_URL'):
    val = os.environ.get(v, '')
    ck(v, bool(val), ('已设置(%d字符)' % len(val)) if val else '缺失 -> rubric 会全空')

print()
print('=== 4. 分片参数 ===')
sn = int(os.environ.get('SHARD_N', 1))
si = int(os.environ.get('SHARD_ID', 0))
ck('SHARD_N/SHARD_ID 合法', sn >= 1 and 0 <= si < sn, 'SHARD_N=%d SHARD_ID=%d' % (sn, si))
ck('多机模式已开启', sn > 1, '单机模式' if sn == 1 else '分片 %d/%d' % (si, sn), hard=False)

print()
print('=== 5. resume 种子 ===')
_HERE = os.path.dirname(os.path.abspath(__file__))
od = os.environ.get('OUTPUT_DIR', os.path.join(_HERE, 'output.e18.kod'))
cand = os.path.join(od, 'e18_candidates.jsonl')
if os.path.exists(cand):
    import json
    import zlib
    n = mine = 0
    with open(cand, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln).get('data_id')
            except Exception:
                continue
            if d:
                n += 1
                if zlib.crc32(str(d).encode()) % sn == si:
                    mine += 1
    ck('种子文件存在', True, '%d 个 id，其中属于本分片 %d 个' % (n, mine))
    ck('种子含本分片的题', mine > 0 or sn == 1,
       '本分片会跳过 %d 题' % mine, hard=False)
else:
    ck('种子文件存在', False,
       '缺 %s -> 会重跑别的机器已做过的题' % cand, hard=False)

print()
print('=== 6. 数据集缓存 ===')
ck('未设 HF_DATASETS_OFFLINE', os.environ.get('HF_DATASETS_OFFLINE', '') not in ('1', 'true'),
   '设了会因本机缓存 config 名带 hash 后缀而加载失败')

print()
if FAIL:
    print('结论: 不可启动 —— 必须先修: %s' % FAIL)
elif WARN:
    print('结论: 可启动，但注意: %s' % WARN)
else:
    print('结论: 全部通过，可以启动')
sys.exit(1 if FAIL else 0)
