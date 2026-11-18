# RolloutResume Gate0 离线验证

本目录只包含 Gate0 探针，不修改生产代码，也不宣称训练收益。

## 验证目标

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\develop\python\python.exe review\gate0\run_gate0.py
```

脚本使用真实的 `StreamingDriver`、`MultiTurnRollout`、`TurnLedger`、`Env` 和
`LocalEnv`。仅对当前主机缺失的 `swift.dev.template_contract` 与 Windows 缺失的
`resource` 安装导入占位，不替代任何被测状态机。

## 通过标准

以下反例必须全部稳定复现：

1. prompt 已被抽取但仍在 in-flight 时，`state_dict()` 只有计数，没有内容和恢复入口。
2. ready group 被 `_next_batch()` 弹出且 `handed` 已递增后，训练提交前崩溃无法重新交付。
3. 新 rollout 实例无法重新 attach live episode view。
4. 新 `LocalEnv` 实例无法续用旧 episode session。
5. `Env.step` 没有 `effect_key`，替换进程会重复同一工具副作用。
6. `TurnLedger` 没有 token、label、old logprob 的导入导出接口。

同时以下可行性必须成立：

1. 在无运行中子进程的 turn boundary，LocalEnv workspace 可以复制到临时目录并原子 rename。
2. 新 LocalEnv session 可以重新绑定复制后的 workspace，并读回原文件。

全部通过时输出 `RolloutResume-Gate0` 的 `ok: true`。若现有 API 已能无损恢复上述状态，
探针会失败，RFC 必须重新审计，不能继续按原方案实施。
