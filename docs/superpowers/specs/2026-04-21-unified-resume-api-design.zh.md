# 统一的 `resume_from_checkpoint` API 设计

## 问题

当前在 `resume_from_ckpt` 分支上的断点续训 API 暴露了两个相似的方法（`load_training_state` 和 `read_training_progress`），难以区分。调用方必须手动编排模型和数据加载器之间的状态恢复，充当组件之间的数据搬运工。此外，Megatron 后端完全没有这些方法，导致 API 表面不对称。

## 设计原则

每个组件负责自身的状态恢复。调用方只负责编排 —— 不在组件之间搬运数据。

## 目标 API

```python
progress = model.resume_from_checkpoint(checkpoint_path)
dataloader.resume_from_checkpoint(progress['consumed_train_samples'])
```

两行代码。两个后端。不再需要 `resume_utils.py` 辅助工具。

## 返回值约定

`model.resume_from_checkpoint()` 返回一个 dict，包含以下确切的键：

```python
{
    'cur_step': int,                    # 优化器步数
    'consumed_train_samples': int,      # 已消耗的总样本数
    'gradient_accumulation_steps': int, # 保存时的 GAS 值
}
```

后端特定的状态（优化器张量、scaler、RNG、mcore 分片状态）在内部恢复，不对外暴露。

## 组件变更

### 1. TwinkleModel 基类 (`src/twinkle/model/base.py`)

添加抽象方法：

```python
@abstractmethod
def resume_from_checkpoint(
    self,
    checkpoint_dir: str,
    *,
    resume_only_model: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    ...
```

参数说明：
- `checkpoint_dir`: 检查点目录的路径。
- `resume_only_model`: 如果为 True，则仅加载权重 —— 跳过优化器/调度器/RNG 的恢复。适用于使用不同优化器配置进行微调的场景。
- `**kwargs`: 后端特定的参数（例如 `adapter_name`）。

### 2. TransformersModel (`src/twinkle/model/transformers/transformers.py`)

删除公共方法：`load_training_state()`、`read_training_progress()`。

保留私有辅助方法：`_save_training_state()`、`_load_optimizer()`、`_load_scaler_state()`、`_load_rng_state()`、`_get_training_rng_state()`。

新实现：

```python
@remote_function()
def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
    adapter_name = kwargs.get('adapter_name', '')

    # 如果检查点包含适配器文件，则加载适配器权重。
    has_adapter = (
        os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
        or os.path.exists(os.path.join(checkpoint_dir, 'adapter_model.bin'))
    )
    if has_adapter:
        self.load(checkpoint_dir, adapter_name=adapter_name)

    # 读取 trainer_state.json。
    trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    # 完整恢复：优化器、调度器、scaler、RNG。
    if not resume_only_model:
        optimizer_group = self._get_optimizer_group(adapter_name)
        self._load_optimizer(checkpoint_dir, optimizer_group, adapter_name)
        self._load_scaler_state(checkpoint_dir)
        self._load_rng_state(checkpoint_dir)
        optimizer_group.cur_step = trainer_state['cur_step']
        optimizer_group.gradient_accumulation_steps = trainer_state['gradient_accumulation_steps']

    return {
        'cur_step': trainer_state['cur_step'],
        'consumed_train_samples': trainer_state['consumed_train_samples'],
        'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
    }
```

全参数训练：权重在模型初始化时加载，因此 `has_adapter` 为 False，`self.load()` 被跳过。仅恢复训练状态。

### 3. MegatronModel (`src/twinkle/model/megatron/megatron.py`)

**save() 变更：** 当 `save_optimizer=True` 时，同时写入 `trainer_state.json`：

```python
if save_optimizer:
    self._save_mcore_optimizer(checkpoint_dir, optimizer_config=optimizer_config, **kwargs)
    trainer_state = {
        'checkpoint_version': 1,
        'cur_step': optimizer_config.cur_step,
        'consumed_train_samples': kwargs.get('consumed_train_samples', 0),
        'gradient_accumulation_steps': optimizer_config.gradient_accumulation_steps,
    }
    state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    if self.device_mesh.rank == 0:
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f, indent=2)
```

**新的 resume_from_checkpoint()：**

```python
@remote_function(dispatch='all')
def resume_from_checkpoint(self, checkpoint_dir, *, resume_only_model=False, **kwargs):
    adapter_name = kwargs.get('adapter_name', self._get_default_group())

    trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)

    self.load(checkpoint_dir, load_optimizer=not resume_only_model,
              adapter_name=adapter_name, **kwargs)

    return {
        'cur_step': trainer_state['cur_step'],
        'consumed_train_samples': trainer_state['consumed_train_samples'],
        'gradient_accumulation_steps': trainer_state['gradient_accumulation_steps'],
    }
```

Megatron 的 `load(load_optimizer=True)` 已经通过 `_load_mcore_optimizer` 恢复了优化器/调度器/RNG/cur_step。`resume_from_checkpoint` 包装器增加了 `trainer_state.json` 的读取，以获取 `consumed_train_samples`。

### 4. 数据加载器 (`src/twinkle/dataloader/dataloader.py`)

新方法：

```python
def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
    self.skip_consumed_samples(consumed_train_samples)
```

`skip_consumed_samples` 保留原样（不更名）以保持向后兼容。`resume_from_checkpoint` 是今后推荐的公共 API。

### 5. 服务端接口 (`src/twinkle/server/model/twinkle_handlers.py`)

- 删除：`/twinkle/load_training_state`、`/twinkle/read_training_progress`
- 新增：`/twinkle/resume_from_checkpoint`，接受 `checkpoint_dir` 和 `resume_only_model` 参数

### 6. 客户端 SDK (`src/twinkle_client/`、`client_tools/client_generator.py`)

- 删除：`load_training_state()`、`read_training_progress()` 客户端方法
- 新增：`resume_from_checkpoint()` 客户端方法

### 7. Cookbook 变更

- 删除 `cookbook/transformers/resume_utils.py` 中的 `resume_from_checkpoint()` 辅助函数（功能现已内置于模型中）
- 更新所有 cookbook 示例以使用新的两行 API

### 8. 文档

更新 `docs/source_en/Components/Model/TransformersModel.md` 及对应的中文文档，以反映新的 API。

## 迁移摘要

| 之前 | 之后 |
|--------|-------|
| `model.load(path)` | `progress = model.resume_from_checkpoint(path)` |
| `model.load_training_state(path)` | （合并到上方） |
| `model.read_training_progress(path)` | `progress = model.resume_from_checkpoint(path, resume_only_model=True)` |
| `dataloader.skip_consumed_samples(n)` | `dataloader.resume_from_checkpoint(n)` |
| `resume_from_checkpoint(model, dataloader, ...)` (cookbook 工具函数) | 两行内联调用 |
