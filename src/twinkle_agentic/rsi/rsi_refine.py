# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 2 — re-analyze each preprocessed trajectory into a STANDARD solving
flow via an injectable teacher API, mark the key rounds, and attach a per-round
reward method.

Input : the subset produced by rsi_prepare.py (rows with a ``messages`` list).
Output: one refined record per trajectory that could be organized:
    {
      "id":     <passthrough id if present>,
      "system": <original system message, kept verbatim (holds tool defs)>,
      "query":  <original first user message, kept verbatim>,
      "tools":  <original tools, kept verbatim>,
      "rounds": [ {intent, type, tool_call, result, code, reward_method} ],
    }
Trajectories the teacher marks unorganizable (e.g. missing tools) are written to
a separate ``*.unorganizable.jsonl`` log and excluded from the standard set.

Design
------
- The teacher is INJECTABLE: any OpenAI-compatible endpoint, chosen at runtime
  via --teacher-model / --teacher-base-url / --teacher-api-key (env fallbacks
  RSI_TEACHER_* then LLM_BACKUP_*). Nothing about the model is hardcoded.
- Key round = a round that carries a tool call OR a code block. The reward
  method is attached automatically by round content:
      tool call present -> 'tool_result'   (executable, verifiable)
      code only         -> 'rubric'         (no executable signal here)
  Concrete reward thresholds are intentionally left unset (decided in step 3).
- Heartbeat rounds are already stripped upstream by MessageNormalizer (step 1),
  so this stage does not re-handle them.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format import SamplingParams
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.utils import get_logger
from twinkle_agentic.protocol.openai import OpenAI

logger = get_logger()

REWARD_TOOL_RESULT = 'tool_result'
REWARD_RUBRIC = 'rubric'

# Runtime prompt is English on purpose: trajectories are predominantly English,
# and mixing languages degrades the teacher. A Chinese rendering was reviewed
# and approved separately.
REORG_SYSTEM = """\
You are given ONE agent trajectory that solves a single task through multiple \
rounds of tool calls. Produce the STANDARD solving flow for this task as \
STRUCTURED JSON, so it can be parsed.

The original system message (which holds the tool definitions) and the original \
user query are kept separately — do NOT rewrite them. Your job is to output only \
the cleaned, correctly-ordered sequence of KEY rounds: the tool calls and their \
results that materially lead to the solution.

Rules:
- Work ONLY from what actually happens in the trajectory and its real tool \
results. Do NOT invent tools, arguments, or results that are not present.
- Remove redundant, failed-and-abandoned, or out-of-order rounds; reorder the \
remaining rounds into the logical order that reaches the solution.
- Preserve each kept round's tool call (name + arguments) and the tool's \
returned result verbatim.
- In each tool_call's "arguments", include ONLY the parameters that were \
actually passed in that call. Do NOT enumerate the tool's full parameter \
schema, and do NOT add keys whose value is null/empty for unused parameters.
- Write it as a FRESH, clean standard procedure. Do NOT say the original was \
wrong, and do NOT reference "previous attempts", "the original code", or "the \
error above".
- If the trajectory cannot be organized into a standard flow (e.g. required \
tools are missing, the task never reaches a solution), output exactly \
{"unorganizable": true, "reason": "<short reason>"} and nothing else.
- Otherwise output ONLY valid JSON in this schema (no prose outside the JSON):
{
  "rounds": [
    {"intent": "<one line>",
     "type": "tool" or "code",
     "tool_call": {"name": "...", "arguments": {...}} or null,
     "result": "<verbatim tool/exec result>",
     "code": "<code text if type==code, else null>"}
  ]
}
"""

_CODE_FENCE_RE = re.compile(r'```')
_JSON_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.IGNORECASE)


def _first_role(messages: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for m in messages:
        if isinstance(m, dict) and m.get('role') == role:
            return m
    return None


def _strip_json_fence(text: str) -> str:
    """Remove a leading ```json / trailing ``` wrapper if the model added one."""
    text = text.strip()
    text = _JSON_FENCE_RE.sub('', text)
    return text.strip()


def _strip_null_args(tool_call: Any) -> Any:
    """Drop arguments whose value is null/empty from a tool_call.

    Backstop for teachers that echo the tool's full parameter schema and pad
    unused params with null: the original calls only pass real args, so a null
    here is invented noise that would break step-3 argument matching. Keys with
    value None or '' are removed; the rest are kept verbatim.
    """
    if not isinstance(tool_call, dict):
        return tool_call
    args = tool_call.get('arguments')
    if isinstance(args, dict):
        tool_call['arguments'] = {k: v for k, v in args.items() if v is not None and v != ''}
    return tool_call


def attach_reward_method(round_obj: Dict[str, Any]) -> str:
    """Decide the reward method from the round's actual content (not the label).

    A round with a tool call is verifiable by its tool result; a code-only round
    has no executable signal here, so it is scored by rubric.
    """
    if round_obj.get('tool_call'):
        return REWARD_TOOL_RESULT
    if round_obj.get('code') or (isinstance(round_obj.get('type'), str) and round_obj['type'] == 'code'):
        return REWARD_RUBRIC
    # Fallback: treat as rubric (no tool call, no code detected).
    return REWARD_RUBRIC


def build_teacher(args) -> OpenAI:
    """Construct the injectable teacher client from CLI/env (nothing hardcoded)."""
    model = args.teacher_model or os.environ.get('RSI_TEACHER_MODEL') or os.environ.get('LLM_BACKUP_MODEL')
    base_url = args.teacher_base_url or os.environ.get('RSI_TEACHER_BASE_URL') or os.environ.get('LLM_BACKUP_BASE_URL')
    api_key = args.teacher_api_key or os.environ.get('RSI_TEACHER_API_KEY') or os.environ.get('LLM_BACKUP_API_KEY')
    if not model:
        raise ValueError('No teacher model given. Pass --teacher-model or set RSI_TEACHER_MODEL/LLM_BACKUP_MODEL.')
    timeout = float(os.environ.get('RSI_TEACHER_TIMEOUT', '120'))
    max_retries = int(os.environ.get('RSI_TEACHER_MAX_RETRIES', '2'))
    return OpenAI(model=model, api_key=api_key, base_url=base_url,
                  client_kwargs={'timeout': timeout, 'max_retries': max_retries})


def reorder_workflow(traj: Dict[str, Any], teacher: OpenAI,
                     sampling_params: SamplingParams) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Ask the teacher to reorganize one trajectory into the standard-flow JSON.

    Returns (parsed_json, None) on success, or (None, reason) when the teacher
    declares it unorganizable or the response is not parseable.
    """
    payload = json.dumps({'messages': traj.get('messages', []), 'tools': traj.get('tools', [])},
                         ensure_ascii=False)
    request = {'messages': [{'role': 'system', 'content': REORG_SYSTEM},
                            {'role': 'user', 'content': payload}]}
    message = teacher(request, sampling_params)
    if isinstance(message, list):
        message = message[0] if message else {}
    content = message.get('content', '') if isinstance(message, dict) else ''
    if not content.strip():
        return None, 'empty_teacher_response'
    try:
        parsed = json.loads(_strip_json_fence(content))
    except (ValueError, TypeError):
        return None, 'unparseable_json'
    if parsed.get('unorganizable'):
        return None, f"unorganizable:{parsed.get('reason', '')}"
    if not isinstance(parsed.get('rounds'), list) or not parsed['rounds']:
        return None, 'no_rounds'
    return parsed, None


def refine_one(traj: Dict[str, Any], teacher: OpenAI,
               sampling_params: SamplingParams) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Turn one raw trajectory into a standard-flow record (or a drop reason)."""
    messages = traj.get('messages') or []
    system_msg = _first_role(messages, 'system')
    query_msg = _first_role(messages, 'user')
    if query_msg is None:
        return None, 'no_user_query'

    parsed, reason = reorder_workflow(traj, teacher, sampling_params)
    if parsed is None:
        return None, reason

    rounds = []
    for r in parsed['rounds']:
        if not isinstance(r, dict):
            continue
        r = dict(r)
        if r.get('tool_call'):
            r['tool_call'] = _strip_null_args(r['tool_call'])
        r['reward_method'] = attach_reward_method(r)
        rounds.append(r)
    if not rounds:
        return None, 'no_valid_rounds'

    record = {
        'id': traj.get('id'),
        'system': system_msg,
        'query': query_msg,
        'tools': traj.get('tools', []),
        'rounds': rounds,
    }
    return record, None


def main():
    parser = argparse.ArgumentParser(description='RSI step 2: refine trajectories into standard solving flows.')
    parser.add_argument('--input', default='output/rsi/subset.jsonl', help='Subset from rsi_prepare.py.')
    parser.add_argument('--output', default='output/rsi/standard_flows.jsonl', help='Refined standard-flow records.')
    parser.add_argument('--teacher-model', default='', help='Teacher model id (or RSI_TEACHER_MODEL/LLM_BACKUP_MODEL).')
    parser.add_argument('--teacher-base-url', default='', help='Teacher endpoint base url.')
    parser.add_argument('--teacher-api-key', default='', help='Teacher API key.')
    parser.add_argument('--max-workers', type=int, default=int(os.environ.get('RSI_MAX_WORKERS', '8')),
                        help='Concurrent teacher API calls.')
    # Generation knobs (defaults shown; override to taste). Low temperature keeps
    # the reformat deterministic; max-tokens bounds the JSON output size.
    parser.add_argument('--temperature', type=float, default=float(os.environ.get('RSI_TEACHER_TEMPERATURE', '0.0')))
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('RSI_TEACHER_MAX_TOKENS', '8192')))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    unorg_path = os.path.splitext(args.output)[0] + '.unorganizable.jsonl'

    teacher = build_teacher(args)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, num_samples=1,
                                     temperature=args.temperature, top_p=1.0)

    rows = Dataset(DatasetMeta(dataset_id=args.input)).dataset.to_list()
    logger.info(f'[rsi_refine] loaded {len(rows)} trajectories from {args.input}')

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    total = len(rows)
    # Log progress every ~5% (at least every 1) so a long run is not a black box.
    step = max(1, total // 20)
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(refine_one, row, teacher, sampling_params): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                record, reason = fut.result()
            except Exception as e:  # noqa: BLE001
                record, reason = None, f'exception:{type(e).__name__}:{e}'
            if record is not None:
                kept.append(record)
            else:
                dropped.append({'id': row.get('id'), 'reason': reason})
            done += 1
            if done % step == 0 or done == total:
                logger.info(f'[rsi_refine] progress {done}/{total} ({100 * done // total}%) '
                            f'kept={len(kept)} dropped={len(dropped)}')

    # Write JSONL directly (NOT via Dataset.save_as): the Arrow table used by
    # save_as unifies the per-round ``arguments`` struct across all rows, padding
    # every call with the global union of arg names as null (and coercing ints to
    # floats). That corrupts the tool calls, so we serialize each record verbatim.
    with open(args.output, 'w', encoding='utf-8') as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if dropped:
        with open(unorg_path, 'w', encoding='utf-8') as f:
            for d in dropped:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    logger.info(f'[rsi_refine] standard flows: {len(kept)}; dropped/unorganizable: {len(dropped)} '
                f'-> {args.output}' + (f' (+ {unorg_path})' if dropped else ''))


if __name__ == '__main__':
    main()
