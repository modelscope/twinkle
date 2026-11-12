# RSI: worlds, profiles and knowledge-guided curricula

`rsi_grpo.py` proposes tasks, verifies them and trains on the kept proposals.
These additions are opt-in. See [agentic/README.md](agentic/README.md) for the sandbox setup.

## Code layout

The experiment lives in this checkout, outside the installed `twinkle_agentic`
package. `agentic/` contains world tools, profile definitions and seed fixtures,
knowledge-card routing and task history. `RsiChallenger` adds profile-aware
prompts, verifier discrimination checks and novelty rewards using the library's
existing scheduling and rollout lifecycle. Task contracts also provide static
validation for the affinity generator.

Import experiment helpers as `cookbook.rsi.agentic.<module>` from the checkout.
For computed novelty, install `pip install -r cookbook/rsi/requirements.txt`.
Tests live in `tests/cookbook/rsi`; run them with `PYTHONPATH=src python -m pytest tests/cookbook/rsi`.

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
export TWINKLE_RSI_NOVELTY_SOURCE=computed
export TWINKLE_RSI_NOVELTY_FLOOR=0.5
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
| `TWINKLE_TASK_SEEDS` | Accepted task-seed release directory; mutually exclusive with knowledge cards |
| `TWINKLE_KNOWLEDGE_CARDS_SHA256` | Optional expected bank hash |
| `TWINKLE_KNOWLEDGE_SEED` | Deterministic card order, default `0` |
| `TWINKLE_TASK_BANK` | JSONL ledger populated from kept recorder entries |
| `TWINKLE_RSI_NOVELTY_SOURCE` | `computed` (default), `judge`, or `off` |
| `TWINKLE_RSI_NOVELTY_FLOOR` | Novelty reward floor; `1.0` keeps difficulty-only rewards, `0.5` enables the original run setting |
| `TWINKLE_RSI_NOVELTY_QUANTILE` | Sibling cosine quantile for clustering, default `0.75` |
| `TWINKLE_RSI_NOVELTY_MODEL` / `_TRIES` | Judge model override / retries, default `3` tries |
| `TWINKLE_RSI_OVERLONG_BUDGET` / `_BUFFER` / `_MAX_PENALTY` | Generated-token penalty budget / ramp / cap; defaults `0` (disabled) / `4096` / `1.0` |
| `TWINKLE_RSI_OVERLONG_FILTER` | Set `1` to drop truncated trajectories after group advantage calculation; default `0` |
| `TWINKLE_RSI_MAX_BUILD_FILES` | New-file budget stated in the author prompt, default `4`; `0` removes the cap |
| `TWINKLE_RSI_STOP_AFTER_STUCK_TURNS` | Stop repeating ineffective tool calls after `2` turns; `0` disables |
| `TWINKLE_RSI_MAX_TRAJECTORY_TOKENS` | Optional whole-trajectory token limit, default `0` (unset) |

World and knowledge bundles and profile IDs are validated before distributed initialization.
Each slot shares the loaded world data but owns its receipts and database role;
reset discards that episode state. Live search and PostgreSQL require deployment
backends implementing the contracts in `cookbook.rsi.agentic.world`.

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

The affinity table selects a profile from the anchor card for both knowledge
and control groups. Every sibling proposal and its solver attempts use that
profile, including on a different slot. Without a match, the configured rotation
selects one fallback for the whole group. Prompts describe seeded, empty or
host-backed inputs accordingly; a failed switch rejects that job.

Generate the advisory table from an existing schema-1 card bank:

```bash
python -m cookbook.rsi.agentic.classify_sandbox_affinity \
    --bank /data/bank/knowledge_cards.jsonl --output-dir /data/affinities \
    --expected-sha256 <bank-sha256> --expected-count 429
export TWINKLE_TOPIC_AFFINITIES=/data/affinities/affinities.jsonl
```

## Knowledge routing and task history

Knowledge/control groups share an anchor card. Control returns no card, while
other configured seeders still contribute text. Use `--num-keywords 0` and no
`--seed-dataset` when a comparison needs an unseeded control group.

The task ledger stores the selected task from each kept group, with recorder
recovery at startup, each iteration and after the final batch. It supplies
historical references to the judge and a fitting corpus to computed novelty.
`novelty_scores.jsonl` records measurements; trajectory metadata records the
provider route, actual profile, verifier checks and length penalty.

Computed novelty compares sibling statements using TF-IDF distance and cluster
spread, fitted on history plus the current group. The proposer reward is
`difficulty * (floor + (1 - floor) * novelty)`. Unmeasured novelty leaves the
difficulty unchanged; zero solver successes still earn zero. Judge mode uses
the existing rubric API and drops groups left unscored after retries. Configure
its credentials with the existing `LLM_BACKUP_*` variables. When an API writes
the check and statement, those turns are context rather than training targets;
only the policy's own generated tokens contribute to its RL update.

Generated Python checks must declare `DERIVED_OUTPUTS`. Before accepting a task,
the check must pass on the author's workspace, fail on an empty workspace and
fail after mutating at least one declared output. The author state is restored
after these checks. Task-seed releases can replace knowledge cards using
`TWINKLE_TASK_SEEDS`; their sampling epoch advances after optimizer updates.
