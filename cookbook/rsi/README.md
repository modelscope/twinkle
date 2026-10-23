# RSI: optional worlds, profiles and knowledge cards

`rsi_grpo.py` proposes tasks, verifies them and trains on the kept proposals.
These additions are opt-in. See [agentic/README.md](agentic/README.md) for the sandbox setup.

## Example

Use a Qwen3.5 model supported by the entry point's `Qwen3_5Template`.
The world, search corpus and knowledge-card bundle must already exist.

```bash
export SAVE_DIR=/data/rsi-run
export TWINKLE_WORLD_ROOT=/data/world
export TWINKLE_WORLD_SEARCH_ROOT=/data/search
export TWINKLE_RSI_PROFILES=world-forensics@1,web-research@1,sql-analytics@1,docs-retrieval@1
export TWINKLE_KNOWLEDGE_CARDS=/data/bank/knowledge_cards.jsonl
export TWINKLE_TASK_BANK=$SAVE_DIR/tasks.jsonl
python cookbook/rsi/rsi_grpo.py --model-id /data/models/qwen3.5-4b \
    --sandbox-template twinkle-rsi-msagent --sandbox-api-url http://sandbox:8000 \
    --save-dir "$SAVE_DIR" --num-envs 32 --num-keywords 0 \
    --micro-batch-size 1 --max-length 16384
```

Use the existing GPU and training arguments for your deployment. In the tested
4B, 16K-token configuration, micro-batch size 1 was required to fit in memory.
Unset these environment variables for the existing unprofiled workflow.

## Configuration

| Variable | Purpose / default |
| --- | --- |
| `TWINKLE_WORLD_ROOT` | Bundle containing `manifest.json`, `ops.db` and `docs/` |
| `TWINKLE_WORLD_SEARCH_ROOT` | Optional hashed `corpus.json` bundle |
| `TWINKLE_WORLD_SEARCH_BACKEND` | `corpus` by default; another value loads a live backend |
| `TWINKLE_WORLD_SEARCH_MODULE` | Directory containing live `search_backend.py` |
| `TWINKLE_WORLD_DB_MODULE` | Directory containing database `provision.py` |
| `TWINKLE_WORLD_DB_HOST` / `_PORT` / `_LEASE` | `127.0.0.1` / `5432` / `21600` seconds |
| `TWINKLE_RSI_PROFILES` | Comma-separated slot rotation; overrides automatic rotation |
| `TWINKLE_TOPIC_AFFINITIES` | External affinity JSONL; enables the host-supported rotation when no explicit list is set |
| `TWINKLE_KNOWLEDGE_CARDS` | Knowledge-card JSONL with its manifest and required release files |
| `TWINKLE_KNOWLEDGE_CARDS_SHA256` | Optional expected bank hash |
| `TWINKLE_KNOWLEDGE_SEED` | Deterministic card order, default `0` |
| `TWINKLE_TASK_BANK` | JSONL ledger populated from kept recorder entries |

World and knowledge bundles and profile IDs are validated before distributed initialization.
Each slot shares the loaded world data but owns its receipts and database role;
reset discards that episode state. Live search and PostgreSQL require deployment
backends implementing the contracts in `twinkle_agentic.envs.world`.

## Profiles

| Profile (`@1`) | Starting data |
| --- | --- |
| `python-compute`, `code-python` | Empty workspace, dependency checks where configured |
| `sql-analytics`, `data-wrangling` | Seeded SQLite / CSV |
| `docs-retrieval`, `log-forensics`, `repo-patch` | Seeded documents / logs / repository |
| `world-forensics`, `web-research`, `db-analyst` | Host database / documents / retrieval tools |
| `knowledge-answer` | No-tool session |
| `lean-proof` | Requires its unavailable runtime |
| `legacy-agentic` | Retired; rejects new tasks |

`slot.py` composes `ProfiledEnvMixin` before `WorldToolsMixin` so profiles filter
host tools too. On environments exposing only `run_command`, allowing aliases
such as `glob` grants that command tool, including shell execution.

The affinity reader exposes `for_card()` for consumers that need per-card
choices. This cookbook uses its host-supported rotation; it does not switch
a slot based on each card. The offline classifier is not included.

## Knowledge routing and task history

Knowledge/control groups share an anchor card. Control returns no card, while
other configured seeders still contribute text. Use `--num-keywords 0` and no
`--seed-dataset` when a comparison needs an unseeded control group.

The task ledger reads `<save-dir>/trajs/index.jsonl` at startup, each iteration
and after the final batch. Set both `TWINKLE_TASK_BANK` and `--save-dir` to record
kept statements. Ledger references are available to callers; training does not
currently use them to score novelty.

`TaskContract` validates task/profile declarations. `NoveltyScorer` computes
TF-IDF distances and cluster spread; fit it on the run's own statements because
unknown vocabulary is ignored. Both remain library APIs: this cookbook does not
construct contracts or change the challenger reward. Install `scikit-learn` to
use the scorer.
