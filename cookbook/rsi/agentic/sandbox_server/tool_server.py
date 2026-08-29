"""ms-agent's tool runtime, served over HTTP from inside the sandbox.

This is the half of the RSI setup that runs *in* the microVM. It builds a real
``LLMAgent`` from the same ``rsi_agent.yaml`` the training host reads, lets
ms-agent prepare its own tools, and exposes two things over HTTP:

* ``GET /tools`` -- the tool schemas, taken from the runtime that will execute
  them. The training host advertises these to the model verbatim, so the
  contract in the prompt and the code behind it cannot drift apart.
* ``POST /call`` -- dispatch, through ms-agent's own ``single_call_tool`` /
  ``parallel_call_tool``.

Nothing here reimplements a tool. That is the whole point: the policy is
trained against the same ``edit_file`` / ``grep`` / ``shell_executor`` behaviour
it will meet at serving time, down to the output formatting. A reimplementation
would be cheaper, but in RL any divergence gets actively exploited by the policy
and only shows up after deployment.

The server is deliberately stdlib-only so the sandbox image stays close to
ms-agent's own dependency set.

Run inside the sandbox::

    python tool_server.py --config /opt/rsi/rsi_agent.yaml --workspace /workspace
"""
import argparse
import asyncio
import copy
import inspect
import json
import os
import sys
import threading
import traceback
from concurrent.futures import TimeoutError as FuturesTimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_PORT = 8900

# read_file's only LLM-backed argument: it summarises a file instead of
# returning it verbatim. Without a reachable LLM the tool cannot honour it, so
# it is also removed from the advertised schema -- see `_usable_llm`.
_LLM_BACKED_ARGS = {'file_system---read_file': ('abbreviate', )}

_SINGLE_NS_FLAG = '_twinkle_single_namespace'

# Marks a permission function this file has already replaced, so a second
# ToolRuntime in one process does not wrap a wrapper.
_PERMISSION_FLAG = '_twinkle_permission_relaxed'

# ms-agent namespaces every tool as ``{server}---{tool}``.
_TOOL_SPLIT = '---'

# Arguments that belong to ms-agent's plumbing rather than to any one tool, and
# that it invites the model to pass without every tool accepting one. Its own
# timeout message says to "set numeric field 'timeout' in the tool arguments"
# (tool_manager.py:687), but only the code_executor trio has a ``timeout``
# parameter, so following that advice on write_file raises TypeError;
# ``description`` is documentation that two of those three declare and the third
# does not; the call id is injected by the host. For a tool whose signature has
# no room for one of these, it is dropped -- the alternative is failing a call
# ms-agent itself asked for. Measured over 5793 calls: 11 ``timeout`` on
# file_system tools, 2 ``description`` on shell_executor.
_FRAMEWORK_ARGS = ('timeout', 'description', 'call_id', '__call_id')

# Withdrawn from the advertised schema whatever ms-agent declares: ``__call_id``
# is a correlation id the host injects ("injected by host when supported",
# local_code_executor.py:494). Advertising it puts an internal handle in the
# prompt and invites the model to invent values for it.
_INTERNAL_ARGS = ('__call_id', )


def _single_namespace_source(code: str) -> str:
    """Wrap ``code`` so it runs in one namespace and cannot exit the process.

    The inner ``exec`` passes one dict twice, which is what ordinary module
    execution does, so nested scopes see top-level names; and ``SystemExit`` /
    ``KeyboardInterrupt`` are turned into stderr text -- which is what ms-agent
    reads as ``success: false`` -- instead of escaping into this server's event
    loop. Stdout written before the exit survives, and ``sys.exit(0)`` stays a
    success. ``repr`` handles the quoting, so the original source survives byte
    for byte.
    """
    return ('import builtins as _tw_builtins\n'
            'import sys as _tw_sys\n'
            '_tw_src = ' + repr(code) + '\n'
            "_tw_ns = {'__name__': '__main__', '__builtins__': _tw_builtins}\n"
            'try:\n'
            "    exec(compile(_tw_src, '<tool>', 'exec'), _tw_ns, _tw_ns)\n"
            'except (SystemExit, KeyboardInterrupt) as _tw_exit:\n'
            "    _tw_status = getattr(_tw_exit, 'code', 1)\n"
            '    if _tw_status not in (0, None):\n'
            "        _tw_sys.stderr.write('%s: %s\\n' % (type(_tw_exit).__name__, _tw_status))\n")


def _patch_python_executor() -> bool:
    """Give ms-agent's local ``python_executor`` ordinary module semantics.

    ``LocalCodeExecutionTool.python_executor`` calls
    ``exec(code, globals_dict, locals_dict)`` with two *different* dicts
    (ms_agent/tools/code/local_code_executor.py:670). Python then runs the code
    the way it runs a class body: top-level assignments land in ``locals_dict``,
    but every nested scope -- a function body, a generator expression --
    resolves free names against ``globals_dict`` alone. So::

        import os
        paths = ['a.txt']
        assert all(os.path.exists(p) for p in paths)

    raises ``NameError: name 'os' is not defined``, which reads as if the model
    wrote broken code. Here it is worse than noise: check scripts arrive through
    this tool and are the reward's ground truth, so a correct check scores as a
    failure.

    The same method catches only ``Exception``, so ``sys.exit(3)`` in a script
    raises ``SystemExit`` out of its ``asyncio.to_thread`` call. ``asyncio.Task``
    re-raises that one after storing it, so it unwinds ``run_forever`` and kills
    :class:`_LoopThread` -- after which every later tool call in the sandbox
    waits for a loop that is gone. Verified: without this, a ``sys.exit(3)``
    call is followed by timeouts on scripts that passed moments earlier.

    And because that ``exec`` runs in *this* process, a relative path in model
    code resolves against this process's cwd -- not against the workspace that
    every other tool uses (``shell_executor`` and ``file_system`` both pass
    ``cwd=self._ws.root`` explicitly). Measured in a live sandbox before this
    chdir: ``write_file 'a.txt'`` answered "Save file successfully", the next
    python call got ``[Errno 2] No such file or directory: 'a.txt'``, and the file
    was in ``/workspace`` while python looked in ``/``. It cost 41 of ex7's 58
    such failures, and files python did write landed outside the directory the
    end-of-episode snapshot lists, so they were invisible to whoever writes the
    check script. The chdir is per call rather than once at startup so that this
    holds however the server was launched; it is the same directory every time,
    so concurrent calls in one turn cannot pull each other around.

    The chdir alone does not let ``import`` find a module the model wrote:
    ``import`` searches ``sys.path``, which holds the server's launch dir
    (``/opt/rsi``), not the workspace and not ``''``. Measured in a live sandbox:
    after ``write_file 'mymod.py'``, ``open('rel.txt')`` read fine but
    ``import mymod`` raised ``ModuleNotFoundError``, so the natural "write a
    helper .py then import and run it" loop failed every time. So the workspace
    is put on ``sys.path`` too, kept as the first entry and never duplicated.

    Duplicated from ``twinkle_agentic.harness.ms_agent`` on purpose -- this file
    is uploaded into a sandbox that has ms-agent and nothing else. Temporary,
    pending an upstream PR.
    """
    from ms_agent.tools.code.local_code_executor import LocalCodeExecutionTool

    original = LocalCodeExecutionTool.python_executor
    if getattr(original, _SINGLE_NS_FLAG, False):
        return False

    async def python_executor(self, code, description='', timeout=None):
        root = getattr(self, 'output_dir', None) or getattr(getattr(self, '_ws', None), 'root', None)
        if root:
            os.makedirs(root, exist_ok=True)
            os.chdir(root)
            # So ``import`` finds a module the model just wrote here. chdir moves
            # cwd but not the import search path, and the workspace is not on it.
            # Must be a str: the import machinery's path finders ignore a
            # PathLike entry on sys.path, and ``root`` arrives as a PosixPath.
            root_str = os.fspath(root)
            if sys.path[:1] != [root_str]:
                if root_str in sys.path:
                    sys.path.remove(root_str)
                sys.path.insert(0, root_str)
        return await original(self, _single_namespace_source(code),
                              description=description, timeout=timeout)

    setattr(python_executor, _SINGLE_NS_FLAG, True)
    LocalCodeExecutionTool.python_executor = python_executor
    return True


def _patch_permission(unrestricted_removal: bool,
                      allow_write_globs: bool) -> List[str]:
    """Honour two safety switches ms-agent's config schema does not implement.

    ``rsi_agent.yaml`` asks for ``safety_rules.unrestricted_removal`` and
    ``safety_rules.allow_write_globs``. ``SafetyConfig.from_dict`` reads only the
    keys it knows and ignores the rest without a word, so on an unmodified
    ms-agent both are dead letters: the sandbox goes on refusing ``rm -rf
    build/*``, ``cp src/* dst/`` and ``chmod +x bin/*``, and the only symptom is
    a run whose tasks are quietly narrower than the config asked for.

    Done as a runtime patch rather than by editing ms-agent, because ms-agent is
    a supported harness and an ordinary dependency: a forked ``permission``
    package would have to be carried, and re-merged, by everyone who runs this
    cookbook. Same reasoning and same shape as :func:`_patch_python_executor`.

    Both refusals are written in ``path_validator``, but ``shell_validator`` and
    ``safety`` pulled them into their own namespaces with ``from ... import``,
    so the replacement is written into every module holding a reference --
    patching the source module alone would leave the copies that actually get
    called untouched.

    Returns the names of the patches applied, for the startup line.
    """
    from ms_agent.permission import path_validator, safety, shell_validator

    applied: List[str] = []
    # Every module that holds a reference, source module included.
    targets = (path_validator, shell_validator, safety)

    if unrestricted_removal:
        original_removal = path_validator.is_dangerous_removal_path
        # Skips this block only, never the one below: an early return here left
        # allow_write_globs unapplied whenever the two were configured together.
        if not getattr(original_removal, _PERMISSION_FLAG, False):

            def is_dangerous_removal_path(path, extra_patterns=(), *args, **kwargs):
                """No path is too dangerous to remove inside a disposable microVM.

                A blanket bypass, including ``dangerous_removal_paths``: the
                checks this switch exists to drop are the fixed ones -- ``*``,
                anything ending in ``/*``, ``/``, a direct child of ``/`` (which
                ``/workspace`` is) and the home directory -- and they are
                entangled with the configurable list in one function. Honouring
                the list here would mean restating ms-agent's matching rules,
                which is the duplication this whole approach avoids. The caller
                is warned at startup when it configured a list this makes moot.
                """
                return False

            setattr(is_dangerous_removal_path, _PERMISSION_FLAG, True)
            for module in targets:
                if hasattr(module, 'is_dangerous_removal_path'):
                    module.is_dangerous_removal_path = is_dangerous_removal_path
            applied.append('unrestricted_removal')

    if allow_write_globs:
        original_validate = path_validator.validate_path
        if not getattr(original_validate, _PERMISSION_FLAG, False):

            def validate_path(path, cwd, allowed_dirs, op_type, **kwargs):
                """Let a glob through a write/create path, scope-checked as usual.

                The glob is handed on as the directory it expands inside, which
                is what the original checks anyway once past the deny -- and it
                is ms-agent's own ``get_glob_base_directory`` that decides where
                that boundary falls, so no policy is restated here. Quotes are
                stripped first for the same reason the original does it: a
                quoted ``'src/*'`` would otherwise yield a base of ``'src``.
                """
                if op_type in ('write', 'create'):
                    bare = path
                    if len(bare) >= 2 and bare[0] == bare[-1] and bare[0] in ('"', "'"):
                        bare = bare[1:-1]
                    if path_validator.GLOB_CHARS & set(bare):
                        base = path_validator.get_glob_base_directory(bare)
                        return original_validate(base, cwd, allowed_dirs, op_type,
                                                 **kwargs)
                return original_validate(path, cwd, allowed_dirs, op_type, **kwargs)

            setattr(validate_path, _PERMISSION_FLAG, True)
            for module in targets:
                if hasattr(module, 'validate_path'):
                    module.validate_path = validate_path
            applied.append('allow_write_globs')

    return applied


def _usable_llm(cfg) -> bool:
    """Whether the declared ``llm`` section can actually serve a request.

    Call this on the config *after* ``LLMAgent`` construction. ms-agent merges
    its own ``agent.yaml`` underneath the user's, and that default declares
    ``service: modelscope``. So an absent ``llm:`` section in rsi_agent.yaml does
    not mean "no LLM" -- it means "modelscope, with no credentials", which
    asserts as soon as FileSystemTool is constructed. The presence of a key is
    what decides it.
    """
    llm = getattr(cfg, 'llm', None)
    if llm is None:
        return False
    service = str(getattr(llm, 'service', '') or '')
    key_fields = (f'{service}_api_key', 'api_key', 'openai_api_key')
    return any(getattr(llm, f, None) or os.environ.get(f.upper()) for f in key_fields)


def _to_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one ms-agent tool schema to the OpenAI shape.

    ``ToolManager.get_tools`` yields ms-agent's own flat form --
    ``{tool_name, server_name, description, parameters}`` -- but the schemas
    served here go into the policy's prompt and are also what
    ``RemoteMsAgentToolEnv.tool_names`` reads, and both speak OpenAI's nested
    ``{type: function, function: {...}}``. Converting at this boundary keeps
    ``/tools`` in the one shape every consumer expects.

    This mirrors ``twinkle_agentic.harness.ms_agent._ms_tools_to_openai``, which
    cannot be imported: this file is uploaded into a sandbox that has ms-agent
    and nothing else.
    """
    if schema.get('type') == 'function' and isinstance(schema.get('function'), dict):
        return schema
    name = schema.get('tool_name') or schema.get('name')
    if not name:
        return schema
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': schema.get('description', ''),
            'parameters': schema.get('parameters') or {'type': 'object', 'properties': {}},
        },
    }


def _without_llm_args(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop arguments this deployment cannot serve from a tool schema.

    Everything reachable from ``/tools`` has to be executable, or the model
    spends the episode learning that a documented argument is broken and carries
    that lesson to a deployment where it works.
    """
    fn = schema.get('function') or {}
    drop = _LLM_BACKED_ARGS.get(fn.get('name'))
    properties = ((fn.get('parameters') or {}).get('properties') or {})
    if not drop or not any(arg in properties for arg in drop):
        return schema
    schema = copy.deepcopy(schema)
    for arg in drop:
        schema['function']['parameters']['properties'].pop(arg, None)
    return schema


def _without_internal_args(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop arguments the host owns from a tool schema.

    Unlike :func:`_without_llm_args` this does not depend on the deployment:
    ``__call_id`` is never something the model should be choosing, however the
    sandbox is configured.
    """
    fn = schema.get('function') or {}
    parameters = fn.get('parameters') or {}
    properties = parameters.get('properties') or {}
    if not any(arg in properties for arg in _INTERNAL_ARGS):
        return schema
    schema = copy.deepcopy(schema)
    parameters = schema['function']['parameters']
    for arg in _INTERNAL_ARGS:
        parameters['properties'].pop(arg, None)
        if isinstance(parameters.get('required'), list) and arg in parameters['required']:
            parameters['required'].remove(arg)
    return schema


class _LoopThread:
    """A single long-lived asyncio loop, owned by a background thread.

    ms-agent's tools bind state to the loop that created them: the notebook
    kernel, MCP client sessions and subprocess transports all hold references
    to it. Running ``asyncio.run`` per request would strand that state -- the
    notebook would lose its variables between turns -- so one loop is created
    at startup and every request is submitted onto it.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name='ms-agent-loop', daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


class ToolRuntime:
    """Owns the ms-agent agent and answers tool queries against it."""

    def __init__(self, config_path: str, workspace: str) -> None:
        from omegaconf import OmegaConf, open_dict

        from ms_agent.agent.llm_agent import LLMAgent

        # Before any tool is constructed: the patch replaces a method on
        # LocalCodeExecutionTool, and prepare_tools() instantiates it.
        _patch_python_executor()

        cfg = OmegaConf.load(config_path)
        with open_dict(cfg):
            cfg.output_dir = workspace
            # Same non-interactive stubs MsAgentHarness applies on the training
            # host: nothing here can answer a TUI permission prompt, and a tool
            # blocking on stdin would hang the episode until the sandbox timeout.
            cfg.interactive = False
            cfg.permission_mode = 'auto'
        self.agent = LLMAgent(cfg)
        # The llm decision has to be made on the *merged* config, after LLMAgent
        # has layered ms-agent's own agent.yaml underneath ours. Popping `llm`
        # from the pre-merge config only removes our section and lets the
        # default's `service: modelscope` show through, which asserts on the
        # missing key as soon as FileSystemTool is constructed. This mirrors
        # MsAgentHarness._apply_rl_stubs, which mutates agent.config for the same
        # reason.
        with open_dict(self.agent.config):
            self.has_llm = _usable_llm(self.agent.config)
            if not self.has_llm:
                # Leaving an unusable section in place is not an option:
                # FileSystemTool builds a client from it eagerly and asserts on
                # the missing key, so no tool at all would come up.
                self.agent.config.pop('llm', None)
        self.agent._interactive = False
        self.agent._event_sink = None
        self.agent._input_source = None
        self.workspace = workspace
        # Before prepare_runtime(), which is what builds SafetyGuard and its
        # validators. Read off the merged config for the same reason `llm` is:
        # ms-agent layers its own agent.yaml underneath ours, so this is what
        # actually took effect rather than what our file happens to say.
        self.permission_patches = _patch_permission(*self._safety_switches())
        self._loop = _LoopThread()
        self._loop.run(self._prepare())
        # Only after prepare_tools(): a contract can only be read off a tool that
        # exists.
        self._contracts = self._build_contracts()

    async def _prepare(self) -> None:
        self.agent.prepare_runtime()
        await self.agent.prepare_tools()

    def _safety_switches(self) -> Tuple[bool, bool]:
        """``(unrestricted_removal, allow_write_globs)`` as configured.

        Absent means off, which is what an unmodified ms-agent does with these
        keys anyway -- so a config that never mentions them keeps every refusal.

        Warns when ``dangerous_removal_paths`` is configured alongside
        ``unrestricted_removal``, because the patch makes that list moot and a
        silently ignored blacklist is the one outcome worth shouting about.
        """
        rules = {}
        permission = getattr(self.agent.config, 'permission', None)
        if permission is not None:
            rules = getattr(permission, 'safety_rules', None) or {}
        unrestricted = bool(_cfg_get(rules, 'unrestricted_removal', False))
        globs = bool(_cfg_get(rules, 'allow_write_globs', False))
        if unrestricted and _cfg_get(rules, 'dangerous_removal_paths', None):
            sys.stderr.write('[tool_server] WARNING unrestricted_removal bypasses the '
                             'rm/rmdir path check entirely, so the configured '
                             'dangerous_removal_paths list will not be consulted\n')
            sys.stderr.flush()
        return unrestricted, globs

    @property
    def _tm(self):
        return self.agent.tool_manager

    def tools(self) -> List[Dict[str, Any]]:
        """Tool schemas, flattened to a plain OpenAI-shaped list.

        ``get_tools`` groups by server; the model only ever sees the flat list,
        and the names are already namespaced as ``{server}---{tool}``.
        """
        raw = self._loop.run(self._tm.get_tools())
        if isinstance(raw, dict):
            flat: List[Any] = []
            for value in raw.values():
                flat.extend(value if isinstance(value, list) else [value])
        else:
            flat = list(raw or [])
        schemas = [_to_openai(t) for t in flat if isinstance(t, dict)]
        schemas = [_without_internal_args(t) for t in schemas]
        return [t if self.has_llm else _without_llm_args(t) for t in schemas]

    def _build_contracts(self) -> Dict[str, Tuple[Set[str], Optional[Set[str]]]]:
        """Per tool: the arguments advertised, and the ones the code will take.

        Both halves are needed because ms-agent lets them disagree, and every
        disagreement is a call the model was invited to make and cannot. The
        advertised half comes from :meth:`tools`, so it is the exact contract the
        prompt carries; the other from the signature of the method
        ``call_tool`` will ``getattr`` and splat the arguments into
        (filesystem_tool.py:387, local_code_executor.py:583). ``None`` means the
        method takes ``**kwargs`` or could not be introspected -- then nothing is
        assumed and nothing is removed.

        Drift is reported at startup rather than waited for: the last one
        (``shell_executor`` advertising nothing about ``description`` while its
        siblings declare it) cost two calls in 239 before anyone noticed, and it
        was found by reading a trajectory.
        """
        contracts: Dict[str, Tuple[Set[str], Optional[Set[str]]]] = {}
        for schema in self.tools():
            fn = schema.get('function') or {}
            name = fn.get('name')
            if not name:
                continue
            declared = set((fn.get('parameters') or {}).get('properties') or {})
            contracts[name] = (declared, self._accepted_args(name))
        drift = {
            name: sorted(declared - accepted)
            for name, (declared, accepted) in contracts.items()
            if accepted is not None and declared - accepted
        }
        if drift:
            sys.stderr.write('[tool_server] WARNING advertised arguments the implementation '
                             'rejects (dropped at dispatch, fix upstream): %s\n' % (drift, ))
            sys.stderr.flush()
        return contracts

    def _accepted_args(self, name: str) -> Optional[Set[str]]:
        """Keyword names the implementation behind ``name`` accepts, or None."""
        try:
            tool_ins = self._tm._tool_index[name][0]
            method = getattr(tool_ins, name.split(_TOOL_SPLIT)[-1])
            sig = inspect.signature(method)
        except Exception:  # noqa -- an un-introspectable tool just gets no repairs
            return None
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return None
        return {
            n
            for n, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }

    def _reconcile(self, name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Repair what ms-agent's contract breaks; refuse the rest, precisely.

        Two different failures arrive as the same TypeError, and they do not
        deserve the same treatment:

        * an argument ms-agent asked for and cannot take -- its own plumbing
          (:data:`_FRAMEWORK_ARGS`) or a schema that overstates the code -- is
          removed. The model followed the contract it was given; failing the call
          would only teach it to distrust a correct one.
        * an argument the model invented is refused, with the accepted list and
          the tool those arguments actually belong to. Measured over 5793 calls,
          259 ``write_file`` calls carried ``old_string``/``new_string``, which
          are ``edit_file``'s. Rewriting those into a ``content=`` write would
          hide a mistake the model should be trained out of, and would teach it a
          call shape that fails outside this sandbox.

        Returns ``(arguments, error)``; ``error`` is not None when the call must
        not run.
        """
        contract = self._contracts.get(name)
        if contract is None:
            return args, None
        declared, accepted = contract
        args = dict(args)
        for arg in list(args):
            if accepted is None or arg in accepted:
                continue
            if arg in _FRAMEWORK_ARGS or arg in declared:
                args.pop(arg)
        # glob's own default for ``path`` is '' (filesystem_tool.py:392 advertises
        # it as optional), but ms-agent's safety guard rejects an empty file path
        # before dispatch, so a model that spells the default out loud gets
        # "Blocked by safety policy: Empty file path" -- 66 times in 5793 calls.
        # '.' is what '' resolves to once inside the tool.
        if name.split(_TOOL_SPLIT)[-1] == 'glob' and 'path' in args \
                and not str(args.get('path') or '').strip():
            args['path'] = '.'
        unknown = sorted(set(args) - declared)
        if unknown:
            return args, self._argument_error(name, unknown, declared)
        return args, None

    def _argument_error(self, name: str, unknown: List[str], declared: Set[str]) -> str:
        """Say what was rejected, what is accepted, and who owns the rest.

        The last part is the useful one and it costs nothing: the arguments of
        every other advertised tool are already known here, so an argument
        belonging to a sibling can be named as such instead of leaving the model
        to guess which of eleven tools it meant.
        """
        owners: Dict[str, List[str]] = {}
        for other, (other_declared, _accepted) in self._contracts.items():
            if other == name:
                continue
            for arg in unknown:
                if arg in other_declared:
                    owners.setdefault(arg, []).append(other)
        quoted = ', '.join(repr(a) for a in unknown)
        parts = ['Error: %s has no argument %s.' % (name, quoted),
                 'It accepts: %s.' % (', '.join(sorted(declared)) or '(none)')]
        for arg, tools in sorted(owners.items()):
            parts.append('%r belongs to %s.' % (arg, ' or '.join(sorted(tools))))
        parts.append('Re-issue the call with this tool\'s arguments, or call the tool '
                     'the arguments belong to.')
        return ' '.join(parts)

    def call(self, calls: List[Dict[str, Any]], timeout: Optional[float]) -> List[Dict[str, Any]]:
        """Dispatch a turn's tool calls, mirroring how ms-agent itself does it.

        A single call goes through ``single_call_tool`` and a batch through
        ``parallel_call_tool``, matching LLMAgent, so concurrency-sensitive
        tools behave in training exactly as they do in production. Each call is
        put through :meth:`_reconcile` first, and one that cannot run is answered
        from here without reaching ms-agent -- so a batch keeps its shape and
        result *i* still answers call *i*.
        """
        out: List[Optional[Dict[str, Any]]] = [None] * len(calls)
        prepared: List[Tuple[int, Dict[str, Any]]] = []
        for i, c in enumerate(calls):
            name = c.get('tool_name')
            args = c.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args or '{}')
                except ValueError:
                    # ms-agent has its own message for unparseable arguments, and
                    # it names the offending text; leave the call to it.
                    prepared.append((i, {'tool_name': name, 'arguments': c.get('arguments')}))
                    continue
            if not isinstance(args, dict):
                args = {}
            args, error = self._reconcile(name, args)
            if error:
                out[i] = {'observation': error, 'ok': False}
            else:
                prepared.append((i, {'tool_name': name, 'arguments': args}))
        if prepared:
            payload = [p for _i, p in prepared]
            try:
                if len(payload) == 1:
                    results = [self._loop.run(self._tm.single_call_tool(payload[0]), timeout)]
                else:
                    results = self._loop.run(self._tm.parallel_call_tool(payload), timeout)
            except Exception as e:  # noqa
                # One failing tool must not take down the server: the episode can
                # still recover, and a dead server would fail every later step of
                # every trajectory sharing this sandbox.
                #
                # A timeout is spelled out rather than reported as its exception
                # name. `concurrent.futures.TimeoutError` carries no message at
                # all, so the model used to read "Tool call failed. TimeoutError:"
                # -- which says nothing about what to do differently. What it
                # needs to know is that the call was abandoned rather than
                # rejected, that whatever it started may still be running (this
                # cannot cancel a subprocess ms-agent has already spawned), and
                # that a long-running command has somewhere else to go. ex8's
                # episode 23 started an HTTP server in the foreground and stalled
                # the whole turn.
                if isinstance(e, (FuturesTimeoutError, asyncio.TimeoutError)):
                    detail = (f'Timed out: this turn\'s tool calls did not finish within '
                              f'{timeout}s and were abandoned. Whatever they started may '
                              f'still be running. A command that does not return on its own '
                              f'-- a server, a watcher, an interactive program -- has to be '
                              f'started with run_in_background=true, or given an explicit '
                              f'time limit inside the command itself.')
                else:
                    detail = f'Tool call failed. {type(e).__name__}: {e}'
                for i, _p in prepared:
                    out[i] = {'observation': detail, 'ok': False}
            else:
                for (i, _p), r in zip(prepared, list(results)):
                    out[i] = {'observation': _with_timeout_advice(_as_text(r)), 'ok': True}
        return [o if o is not None else {'observation': '', 'ok': False} for o in out]


# ms-agent's own words when its per-call wait runs out (tool_manager.py:687).
_MS_TIMEOUT_MARK = 'Tool call timed out after'
# Appended to it, not substituted for it. Its message offers exactly one remedy --
# raise the `timeout` argument -- which is the wrong one for a command that never
# returns at all: ex8's episode 23 started `python -m http.server` in the
# foreground, and no limit up to the 600s ceiling would have helped. shell_executor
# already advertises `run_in_background`, so this names the argument the model
# already has rather than teaching it anything new.
_TIMEOUT_ADVICE = (
    ' A command that does not return on its own -- a server, a watcher, an '
    'interactive program -- will time out at any limit; start it with '
    'run_in_background=true instead, or bound it inside the command itself.')


def _with_timeout_advice(observation: str) -> str:
    if _MS_TIMEOUT_MARK in observation and 'run_in_background' not in observation:
        return observation + _TIMEOUT_ADVICE
    return observation


def _as_text(result: Any) -> str:
    if result is None:
        return ''
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off an OmegaConf node or a plain dict.

    The permission section arrives as a DictConfig when the yaml declares it and
    as a dict when it is assembled in code, and only one of those answers to
    ``.get``.
    """
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


class _Handler(BaseHTTPRequestHandler):
    runtime: ToolRuntime = None  # set on the class before the server starts
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.startswith('/health'):
            self._reply(200, {'status': 'ok', 'workspace': self.runtime.workspace})
        elif self.path.startswith('/tools'):
            self._guarded(lambda: {'tools': self.runtime.tools()})
        else:
            self._reply(404, {'error': f'no such endpoint: {self.path}'})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith('/call'):
            self._reply(404, {'error': f'no such endpoint: {self.path}'})
            return
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError as e:
            self._reply(400, {'error': f'malformed request body: {e}'})
            return
        calls = body.get('calls') or []
        if not isinstance(calls, list) or not calls:
            self._reply(400, {'error': "'calls' must be a non-empty list"})
            return
        self._guarded(lambda: {'results': self.runtime.call(calls, body.get('timeout'))})

    def _guarded(self, produce) -> None:
        """Answer with ``produce()``, turning a crash into a 500 with a traceback.

        The client surfaces the body as the observation, so a bug in here shows
        up in the trajectory instead of as an opaque connection reset.
        """
        try:
            self._reply(200, produce())
        except Exception:  # noqa
            self._reply(500, {'error': traceback.format_exc()})

    def _reply(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write('[tool_server] %s\n' % (fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='ms-agent yaml, the same one the training host loads')
    parser.add_argument('--workspace', default='/workspace', help='config.output_dir for this episode')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    runtime = ToolRuntime(args.config, args.workspace)
    _Handler.runtime = runtime
    # Threading, because a turn's tool calls arrive as one request but the
    # health poll must stay answerable while a long shell command runs.
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    names = [(t.get('function') or {}).get('name') for t in runtime.tools()]
    llm_note = 'llm configured' if runtime.has_llm else 'no llm (read_file.abbreviate withdrawn)'
    perm_note = (', permission: ' + '+'.join(runtime.permission_patches)
                 if runtime.permission_patches else '')
    sys.stderr.write(f'[tool_server] ready on {args.host}:{args.port}, {llm_note}'
                     f'{perm_note}, {len(names)} tools: {names}\n')
    sys.stderr.flush()
    server.serve_forever()


if __name__ == '__main__':
    main()
