# SkillProvider 技能系统

技能系统允许 Twinkle Auto 智能体从外部来源（Git 仓库、API、本地文件）动态加载专业知识，并注入到 LLM 的系统提示词中。

## 架构

| 类 | 角色 |
|----|------|
| **Skill** | 持有单个技能名称、内容和来源的数据类 |
| **SkillProvider** | 从数据源获取技能的抽象基类 |
| **SkillManager** | 编排多个 Provider，聚合技能用于提示词注入 |

## Skill 数据类

```python
@dataclasses.dataclass
class Skill:
    name: str       # 简短标识符（通常为文件名去除扩展名）
    content: str    # 完整的 Markdown 内容
    source: str     # Provider 名称 + 相对路径，用于可追溯性
```

## 创建自定义 Provider

继承 `SkillProvider` 并实现 `name` 和 `fetch()`：

```python
from twinkle_client.skills.base import SkillProvider

class MySkillProvider(SkillProvider):

    @property
    def name(self) -> str:
        return 'my-skills'

    async def fetch(self) -> None:
        # 将技能文件下载/克隆到 self.cache_dir
        # 例如：git clone、API 下载、文件拷贝
        ...
```

默认的 `load_skills()` 会扫描 `self.cache_dir` 中的 `.md` 文件（跳过 README、LICENSE 等），返回 `Skill` 对象。

## SkillManager

```python
from twinkle_client.skills.manager import SkillManager

manager = SkillManager()
manager.register(my_provider)
manager.register(another_provider)

# 拉取并加载所有技能
skills = await manager.load_all()

# 格式化为 LLM 系统提示词注入内容
prompt_section = manager.format_for_prompt()
```

### 关键方法

| 方法 | 说明 |
|------|------|
| `register(provider)` | 添加技能 Provider |
| `load_all()` | 从所有 Provider 拉取并加载 |
| `format_for_prompt()` | 将技能渲染为系统提示词格式 |
| `get_skill_names()` | 列出已加载技能名称 |

## 缓存目录

默认缓存在 `~/.cache/twinkle/auto/skills/<provider_name>/`。可通过向 Provider 构造函数传入 `cache_dir` 参数覆盖。
