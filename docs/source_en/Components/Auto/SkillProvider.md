# SkillProvider

The skill system allows Twinkle Auto's agent to dynamically load specialized knowledge from external sources (Git repos, APIs, local files) and inject them into the LLM's system prompt.

## Architecture

| Class | Role |
|-------|------|
| **Skill** | Dataclass holding a single skill's name, content, and source |
| **SkillProvider** | Abstract base class for fetching skills from a source |
| **SkillManager** | Orchestrates multiple providers, aggregates skills for prompt injection |

## Skill Dataclass

```python
@dataclasses.dataclass
class Skill:
    name: str       # Short identifier (typically filename without extension)
    content: str    # Full markdown content
    source: str     # Provider name + relative path for traceability
```

## Creating a Custom Provider

Subclass `SkillProvider` and implement `name` and `fetch()`:

```python
from twinkle_client.skills.base import SkillProvider

class MySkillProvider(SkillProvider):

    @property
    def name(self) -> str:
        return 'my-skills'

    async def fetch(self) -> None:
        # Download/clone skill files to self.cache_dir
        # e.g., git clone, API download, file copy
        ...
```

The default `load_skills()` scans `self.cache_dir` for `.md` files (skipping README, LICENSE, etc.) and returns `Skill` objects.

## SkillManager

```python
from twinkle_client.skills.manager import SkillManager

manager = SkillManager()
manager.register(my_provider)
manager.register(another_provider)

# Fetch and load all skills
skills = await manager.load_all()

# Format for LLM system prompt injection
prompt_section = manager.format_for_prompt()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `register(provider)` | Add a skill provider |
| `load_all()` | Fetch + load from all providers |
| `format_for_prompt()` | Render skills as formatted text for system prompt |
| `get_skill_names()` | List names of loaded skills |

## Cache Directory

By default, skills are cached at `~/.cache/twinkle/auto/skills/<provider_name>/`. Override by passing `cache_dir` to the provider constructor.
