# Copyright (c) ModelScope Contributors. All rights reserved.
"""The sandbox as a resource: N microVMs, each one a slot a job can own.

A slot is owned for as long as a job needs it, because the workspace lives inside
the microVM: from the clear, through every tool call, to the check that runs
against what was left behind. Two jobs sharing a slot would read each other's
files, so the pool hands out whole slots and never sub-divides one.

The transport underneath is ``remote_tool_env.RemoteMsAgentToolEnv``, which is
paired with the in-sandbox runtime in ``sandbox_server/tool_server.py``. The
solver's opening messages come from ``episode.solver_harness``, the same function
``eval.py`` uses, so a task's difficulty here and its pass rate there are measured
against one opening.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from twinkle import get_logger
from twinkle_agentic.envs import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager

from episode import solver_harness  # noqa: I100,I202
from remote_tool_env import RemoteMsAgentToolEnv, tool_payload  # noqa: I100,I202

logger = get_logger()

__all__ = ['Sandbox', 'open_pool', 'close_pool', 'solver_harness',
           'CLEAR_WORKSPACE', 'WORKSPACE_SNAPSHOT']

# Cleared through the python tool, not `rm -rf`: ms-agent's safety policy rejects
# `rm -rf` outright ("Blocked by safety rule"), and it rejects globs in write
# operations, which rules out `find -delete` too. The script asserts the
# directory really is empty, so a future policy change surfaces as a failed reset
# instead of jobs quietly inheriting the previous workspace.
CLEAR_WORKSPACE = '''
import os, shutil
root = {workspace!r}
os.makedirs(root, exist_ok=True)
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)
leftover = os.listdir(root)
assert not leftover, 'workspace not empty after clear: %r' % (leftover,)
'''

# The ground truth the check script is written against. A listing alone is not
# enough: three of the six rejected proposals in the first real run failed on a
# value the model recomputed from its own recollection ("Mean values mismatch")
# rather than read off the file, so the end state has to arrive as content, not
# just as names. Bounded on both axes because this goes into a prompt and a 100k
# artifact would push the trajectory it is read alongside out of the window.
#
# Walks the tree in python rather than shelling out to `find`: the same code then
# decides what is text, what is truncated, and what the budget was spent on,
# which a pipeline of find/head cannot report back.
#
# File bodies go out byte for byte. An earlier version printed `body.rstrip()`,
# which hid trailing newlines while the size column still counted them, so a
# check writer shown an 11-byte file whose content looked 10 characters long
# wrote `content == 'Mean: 63.9'` and the check failed against the very state it
# was written from. The listing is only ground truth if it does not tidy up.
#
# Facts *about* a file go in its header, never after its body. A note printed
# below the content is indistinguishable from content: annotated one file with a
# trailing `(no newline at end of file)` line and the next check script asserted
# the README's content ending in that sentence.
WORKSPACE_SNAPSHOT = '''
import os

root = {workspace!r}
skip = {{'.ms_agent', '__pycache__', '.ipynb_checkpoints', '.git'}}
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        try:
            rows.append((os.path.relpath(path, root), os.path.getsize(path), path))
        except OSError:
            pass
rows.sort()
for rel, size, _ in rows[:{max_files}]:
    print(rel, size)

budget = {total_budget}
for rel, size, path in rows[:{max_files}]:
    if budget <= 0:
        break
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read({per_file} + 1)
    except (OSError, UnicodeDecodeError):
        continue          # binary or unreadable: the listing already names it
    if '\\x00' in text:
        continue
    body = text[:{per_file}]
    budget -= len(body)
    # The trailing-newline count is stated for every file, both ways. Saying it
    # only when it is absent made "this file ends with a newline" invisible, and
    # the check writer then compared exact bytes without one: in ex9 two of the
    # three checks that failed their own verification failed on exactly that --
    # the same reply asserted three files, guessed right on the two marked "no
    # newline at end" and wrong on the unmarked one.
    trailing = len(body) - len(body.rstrip(chr(10)))
    if len(text) > len(body):
        suffix = ' (first {per_file} bytes)'
    elif trailing == 0:
        suffix = ' (no newline at end)'
    else:
        suffix = ' (ends with %d newline character(s))' % trailing
    print()
    print('--- ' + rel + suffix + ' ---')
    print(body, end='')
    if not body.endswith(chr(10)):
        print()
'''

# Seconds to wait before asking a sandbox for its workspace listing a second
# time. 62 of run_clean6's 63 snapshot failures were the sandbox answering 410
# "not proxyable", which is the host having paused it -- worth one more ask,
# since the alternative is throwing the job away.
SNAPSHOT_RETRY_WAIT = 3

# Same idea for the workspace clear. Longer, because what it waits out is
# different: a clear times out when ms-agent's per-call limit expires with the
# delete still running, so the second attempt wants the first one's rmtree to
# have drained rather than to race it.
RESET_RETRY_WAIT = 10


class Sandbox:
    """One slot: clear the workspace, run a script in it, read it back.

    Not thread-safe on purpose. A slot belongs to whoever holds it, and the pool
    hands each one to exactly one worker thread.
    """

    def __init__(self, slot: int, env: RemoteMsAgentToolEnv, schemas: list,
                 *, snapshot_max_files: int, snapshot_per_file: int, snapshot_budget: int):
        self.slot = slot
        self.env = env
        self.workspace = env.workspace
        # The advertised tool contract. Carried on the slot because it goes into
        # the prompt: the schemas a trajectory is built with have to be the ones
        # the slot it runs on will honour.
        self.schemas = schemas
        self._runner = env.runner()
        # The tools carry the env they dispatch into, so this slot's model turns
        # have to go through this slot's manager.
        self.tool_manager = ToolManager(EnvTool.from_schemas(env, schemas))
        self._snapshot_script = WORKSPACE_SNAPSHOT.format(
            workspace=self.workspace, max_files=snapshot_max_files,
            per_file=snapshot_per_file, total_budget=snapshot_budget)
        self._clear_script = CLEAR_WORKSPACE.format(workspace=self.workspace)

    def run(self, script: str) -> Tuple[int, str]:
        """Run a python script in the workspace; returns (exit code, output)."""
        return self._runner(script, 'python')

    def clear(self) -> None:
        """Empty the workspace. Raises rather than returning quietly.

        Every caller depends on a clean start: a silent no-op here means a task
        inherits the previous task's files, which lets a solver pass without doing
        anything and makes the difficulty numbers meaningless.

        This is also the one point where losing the sandbox costs nothing, since
        the workspace is about to be emptied regardless -- so a runtime that went
        away is rebuilt here rather than ending a run with hours behind it. The
        same covers a clear that *fails* on a runtime still answering /health:
        run 'rsi' reached iteration 7 and ended on three clears timing out at
        ms-agent's per-call limit while the sandbox reported itself healthy. So
        the clear is retried, then retried on a deliberately rebuilt sandbox.
        """
        if self.env.ensure_ready():
            self._rebind('runtime was unreachable')
        code, out = self.run(self._clear_script)
        if code != 0:
            logger.warning(f'[sandbox {self.slot}] clear failed (exit {code}), '
                           f'retrying in {RESET_RETRY_WAIT}s: {out[-200:]}')
            time.sleep(RESET_RETRY_WAIT)
            code, out = self.run(self._clear_script)
        if code != 0:
            # Rebuilt rather than retried again: two failures in a row is not the
            # transient this waits out, and a fresh sandbox brings a workspace
            # that is already empty -- which is all this method is asked for.
            logger.warning(f'[sandbox {self.slot}] clear failed twice (exit {code}); '
                           f'rebuilding: {out[-200:]}')
            self.env.reset()
            self.env.n_recoveries += 1
            self._rebind('rebuilt after two failed clears')
            code, out = self.run(self._clear_script)
        if code != 0:
            raise RuntimeError(f'workspace clear failed (exit {code}): {out[-400:]}')

    def snapshot(self) -> Tuple[str, str]:
        """The end state as (listing, error).

        Returned as a bare listing, unwrapped from the tool's JSON envelope: the
        model has to read it as a directory rather than as a tool result, or it
        falls back on what it *believes* it created.

        An empty listing with no error means the workspace really was empty. An
        empty listing with an error means it could not be read, and the two are
        kept apart because a snapshot that says "empty" when it means "I could not
        look" produces tasks whose only true assertion is that nothing happened.
        """
        code, out = self.run(self._snapshot_script)
        if code != 0:
            logger.warning(f'[sandbox {self.slot}] snapshot failed (exit {code}), '
                           f'retrying in {SNAPSHOT_RETRY_WAIT}s: {out[-200:]}')
            time.sleep(SNAPSHOT_RETRY_WAIT)
            code, out = self.run(self._snapshot_script)
        if code != 0:
            return '', f'workspace snapshot failed (exit {code}): {out[-500:]}'
        return tool_payload(out).strip(), ''

    def close(self) -> None:
        self.env.close()

    def _rebind(self, why: str) -> None:
        """Point the runner and the tools at the sandbox behind this env now."""
        self._runner = self.env.runner()
        self.tool_manager = ToolManager(EnvTool.from_schemas(self.env, self.env.tool_schemas()))
        logger.warning(f'[sandbox {self.slot}] rebound ({why})')


def open_pool(
    n: int,
    *,
    template: str,
    api_url: str,
    config_path: str,
    workspace: str,
    sandbox_timeout: int,
    snapshot_max_files: int,
    snapshot_per_file: int,
    snapshot_budget: int,
) -> List[Sandbox]:
    """Boot ``n`` slots and return them ready to use.

    Booted in parallel: each is a microVM taking ~10s, and doing them one after
    another would put minutes in front of every run. The tool schemas are read
    once, off the first slot -- every slot runs the same image, and these go
    straight into the prompt, so reading them n times would only add n chances
    for the prompt to differ between slots.
    """
    if not template:
        raise SystemExit('sandbox template is required (--sandbox-template or AENV_TEMPLATE)')
    if not api_url:
        raise SystemExit('sandbox api url is required (--sandbox-api-url or AENV_API_URL)')

    def _boot(_) -> RemoteMsAgentToolEnv:
        env = RemoteMsAgentToolEnv(template=template, config_path=config_path,
                                   api_url=api_url, workspace=workspace,
                                   sandbox_timeout=sandbox_timeout)
        env.reset()
        return env

    n = max(1, n)
    with ThreadPoolExecutor(max_workers=n) as pool:
        envs = list(pool.map(_boot, range(n)))
    schemas = envs[0].tool_schemas()
    slots = [
        Sandbox(i, env, schemas, snapshot_max_files=snapshot_max_files,
                snapshot_per_file=snapshot_per_file, snapshot_budget=snapshot_budget)
        for i, env in enumerate(envs)
    ]
    logger.info(f'[sandbox] {len(slots)} slot(s) ready, tools: '
                f'{[(s.get("function") or {}).get("name") for s in schemas]}')
    return slots


def close_pool(slots: List[Sandbox]) -> int:
    """Kill every slot; returns how many rebuilds happened over the run.

    Reported rather than dropped: a run whose sandboxes were rebuilt twenty times
    produced its numbers under a different environment than one that was rebuilt
    never, and that is invisible from the output files alone.
    """
    total = sum(getattr(s.env, 'n_recoveries', 0) for s in slots)
    for slot in slots:
        try:
            slot.close()
        except Exception as e:  # noqa # best-effort: the backend evicts on timeout anyway
            logger.warning(f'[sandbox {slot.slot}] close failed: {e}')
    return total
