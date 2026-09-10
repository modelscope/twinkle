"""ms-agent's tool runtime, for ``server.py`` to serve from inside the sandbox.

Builds a real ``LLMAgent`` from the same yaml the training host reads, lets
ms-agent prepare its own tools, and answers two questions about them: what they
are (:meth:`Runtime.tools`) and what one does (:meth:`Runtime.call`, through
ms-agent's own ``single_call_tool`` / ``parallel_call_tool``).

Nothing here reimplements a tool. That is the whole point: the policy is trained
against the same ``edit_file`` / ``grep`` / ``shell_executor`` behaviour it will
meet at serving time, down to the output formatting. A reimplementation would be
cheaper, but in RL any divergence gets actively exploited by the policy and only
shows up after deployment.

ms-agent's naming convention is also this file's business alone: tools arrive
namespaced as ``{server}---{tool}`` and are advertised bare wherever that is
unambiguous, then put back before dispatch -- see :meth:`Runtime._advertise`.

Stdlib-only apart from ms-agent itself, so the sandbox image stays close to
ms-agent's own dependency set.
"""
import asyncio
import copy
import inspect
import json
import os
import sys
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set, Tuple

# read_file's only LLM-backed argument: it summarises a file instead of
# returning it verbatim. Without a reachable LLM the tool cannot honour it, so
# it is also removed from the advertised schema -- see `_usable_llm`.
_LLM_BACKED_ARGS = {'file_system---read_file': ('abbreviate', )}

# Withdrawn from the advertised schema whatever ms-agent declares: ``__call_id``
# is a correlation id the host injects, so advertising it puts an internal handle
# in the prompt and invites the model to invent values for it.
_INTERNAL_ARGS = ('__call_id', )

_SINGLE_NS_FLAG = '_twinkle_single_namespace'

# Marks a permission function this file has already replaced, so a second
# ToolRuntime in one process does not wrap a wrapper.
_PERMISSION_FLAG = '_twinkle_permission_relaxed'

# Arguments that belong to ms-agent's plumbing rather than to any one tool, and
# that it invites the model to pass without every tool accepting one. Its own
# timeout message says to "set numeric field 'timeout' in the tool arguments",
# but only the code_executor trio has a ``timeout`` parameter, so following that
# advice on write_file raises TypeError; ``description`` is documentation that
# two of those three declare and the third does not; the call id is injected by
# the host. For a tool whose signature has no room for one of these, it is
# dropped -- the alternative is failing a call ms-agent itself asked for.
# Measured over 5793 calls: 11 ``timeout`` on file_system tools, 2
# ``description`` on shell_executor.
_FRAMEWORK_ARGS = ('timeout', 'description', 'call_id', '__call_id')

# ms-agent's own words when its per-call wait runs out (tool_manager.py), and the
# advice its message lacks. It offers exactly one remedy -- raise the ``timeout``
# argument -- which is the wrong one for a command that never returns at all:
# ex8's episode 23 started ``python -m http.server`` in the foreground and no
# limit up to the 600s ceiling would have helped. ``shell_executor`` already
# advertises ``run_in_background``, so this names an argument the model has
# rather than teaching it anything new.
_MS_TIMEOUT_MARK = 'Tool call timed out after'
_TIMEOUT_ADVICE = (
    'A command that does not return on its own -- a server, a watcher, an '
    'interactive program -- will time out at any limit; start it with '
    'run_in_background=true instead, or bound it inside the command itself.')


def _single_namespace_source(code: str) -> str:
    """Wrap ``code`` so it runs in one namespace and cannot exit the process.

    The inner ``exec`` passes one dict twice, which is what ordinary module
    execution does, so nested scopes see top-level names; and ``SystemExit`` /
    ``KeyboardInterrupt`` become stderr text -- which ms-agent reads as
    ``success: false`` -- instead of escaping into this server's event loop.
    Stdout written before the exit survives, and ``sys.exit(0)`` stays a success.
    ``repr`` handles the quoting, so the source survives byte for byte.
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

    Four repairs, all for one method (local_code_executor.py:773):

    * it ``exec``s with two *different* dicts (line 786), so the code runs the
      way a class body does: top-level assignments land in the locals, but every
      nested scope resolves free names against the globals alone. ``import os``
      followed by ``all(os.path.exists(p) for p in paths)`` raises ``NameError``,
      reading as if the model wrote broken code. Here that is worse than noise:
      check scripts arrive through this tool and are the reward's ground truth,
      so a correct check scores as a failure.
    * it catches only ``Exception``, so ``sys.exit(3)`` raises ``SystemExit`` out
      of its ``asyncio.to_thread`` call, unwinds ``run_forever`` and kills
      :class:`_LoopThread` -- after which every later tool call waits for a loop
      that is gone. Verified: without this, a ``sys.exit(3)`` call is followed by
      timeouts on scripts that passed moments earlier.
    * that ``exec`` runs in *this* process, so a relative path resolves against
      this process's cwd, not the workspace every other tool passes as ``cwd``.
      Measured live: ``write_file 'a.txt'`` answered "Save file successfully",
      the next python call got ``No such file or directory: 'a.txt'`` -- 41 of
      ex7's 58 such failures, and files python did write landed outside the
      directory the end-of-episode snapshot lists.
    * ``import`` searches ``sys.path``, which chdir does not move, so after
      ``write_file 'mymod.py'`` the natural "write a helper then import it" loop
      raised ``ModuleNotFoundError`` every time. The workspace goes on the path
      too, first entry, never duplicated.

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
            # Must be a str: the import machinery's path finders ignore a
            # PathLike entry on sys.path, and ``root`` arrives as a PosixPath.
            root_str = os.fspath(root)
            if sys.path[:1] != [root_str]:
                if root_str in sys.path:
                    sys.path.remove(root_str)
                sys.path.insert(0, root_str)
        return await original(self, _single_namespace_source(code), description=description, timeout=timeout)

    setattr(python_executor, _SINGLE_NS_FLAG, True)
    LocalCodeExecutionTool.python_executor = python_executor
    return True


def _patch_permission(unrestricted_removal: bool, allow_write_globs: bool) -> List[str]:
    """Honour two safety switches ms-agent's config schema does not implement.

    ``rsi_agent.yaml`` asks for ``safety_rules.unrestricted_removal`` and
    ``safety_rules.allow_write_globs``. ``SafetyConfig.from_dict`` reads only the
    keys it knows and ignores the rest without a word, so on an unmodified
    ms-agent both are dead letters: the sandbox goes on refusing ``rm -rf
    build/*``, ``cp src/* dst/`` and ``chmod +x bin/*``, and the only symptom is a
    run whose tasks are quietly narrower than the config asked for.

    A runtime patch rather than an edit to ms-agent, which is a supported harness
    and an ordinary dependency: a forked ``permission`` package would have to be
    carried, and re-merged, by everyone who runs this cookbook.

    Both refusals live in ``path_validator``, but ``shell_validator`` and
    ``safety`` pulled them into their own namespaces with ``from ... import``, so
    the replacement is written into every module holding a reference -- patching
    the source module alone would leave the copies that actually get called.

    Returns the names of the patches applied, for the startup line.
    """
    from ms_agent.permission import path_validator, safety, shell_validator

    applied: List[str] = []
    targets = (path_validator, shell_validator, safety)

    if unrestricted_removal and not getattr(path_validator.is_dangerous_removal_path, _PERMISSION_FLAG, False):

        def is_dangerous_removal_path(path, extra_patterns=(), *args, **kwargs):
            """No path is too dangerous to remove inside a disposable microVM.

            A blanket bypass, ``dangerous_removal_paths`` included: the checks
            this switch exists to drop are the fixed ones -- ``*``, anything
            ending in ``/*``, ``/``, a direct child of ``/`` (which
            ``/workspace`` is) and the home directory -- and they are entangled
            with the configurable list in one function. Honouring the list here
            would mean restating ms-agent's matching rules, which is the
            duplication this whole approach avoids. The caller is warned at
            startup when it configured a list this makes moot.
            """
            return False

        setattr(is_dangerous_removal_path, _PERMISSION_FLAG, True)
        for module in targets:
            if hasattr(module, 'is_dangerous_removal_path'):
                module.is_dangerous_removal_path = is_dangerous_removal_path
        applied.append('unrestricted_removal')

    if allow_write_globs and not getattr(path_validator.validate_path, _PERMISSION_FLAG, False):
        original_validate = path_validator.validate_path

        def validate_path(path, cwd, allowed_dirs, op_type, **kwargs):
            """Let a glob through a write/create path, scope-checked as usual.

            The glob is handed on as the directory it expands inside, which is
            what the original checks anyway once past the deny -- and it is
            ms-agent's own ``get_glob_base_directory`` that decides where that
            boundary falls, so no policy is restated here. Quotes are stripped
            first for the same reason the original does it: a quoted ``'src/*'``
            would otherwise yield a base of ``'src``.
            """
            if op_type in ('write', 'create'):
                bare = path
                if len(bare) >= 2 and bare[0] == bare[-1] and bare[0] in ('"', "'"):
                    bare = bare[1:-1]
                if path_validator.GLOB_CHARS & set(bare):
                    base = path_validator.get_glob_base_directory(bare)
                    return original_validate(base, cwd, allowed_dirs, op_type, **kwargs)
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
    served here go into the policy's prompt, and both the prompt and the client
    speak OpenAI's nested ``{type: function, function: {...}}``.

    Mirrors ``twinkle_agentic.harness.ms_agent._tool_to_openai``, which cannot be
    imported: this file is uploaded into a sandbox that has ms-agent and nothing
    else.
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
            'parameters': schema.get('parameters') or {
                'type': 'object',
                'properties': {}
            },
        },
    }


def _without_args(schema: Dict[str, Any], names: Tuple[str, ...]) -> Dict[str, Any]:
    """Drop ``names`` from a tool schema's parameters, ``required`` included.

    Everything reachable from ``/tools`` has to be executable, or the model
    spends the episode learning that a documented argument is broken and carries
    that lesson to a deployment where it works.
    """
    parameters = (schema.get('function') or {}).get('parameters') or {}
    if not any(name in (parameters.get('properties') or {}) for name in names):
        return schema
    schema = copy.deepcopy(schema)
    parameters = schema['function']['parameters']
    for name in names:
        parameters['properties'].pop(name, None)
        if isinstance(parameters.get('required'), list) and name in parameters['required']:
            parameters['required'].remove(name)
    return schema


class _LoopThread:
    """A single long-lived asyncio loop, owned by a background thread.

    ms-agent's tools bind state to the loop that created them: the notebook
    kernel, MCP client sessions and subprocess transports all hold references to
    it. Running ``asyncio.run`` per request would strand that state -- the
    notebook would lose its variables between turns -- so one loop is created at
    startup and every request is submitted onto it.
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


class Runtime:
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
            # Same non-interactive settings MsAgentHarness applies on the
            # training host: nothing here can answer a TUI permission prompt,
            # and a tool blocking on stdin would hang the episode until the
            # sandbox timeout. ``permission.mode`` is the nested key
            # PermissionConfig.from_dict reads.
            cfg.interactive = False
            OmegaConf.update(cfg, 'permission.mode', 'auto', merge=True)
        self.agent = LLMAgent(cfg)
        # The llm decision has to be made on the *merged* config, after LLMAgent
        # has layered ms-agent's own agent.yaml underneath ours. Popping `llm`
        # from the pre-merge config only removes our section and lets the
        # default's `service: modelscope` show through, which asserts on the
        # missing key as soon as FileSystemTool is constructed.
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
        # Only after prepare_tools(): a name can only be read off a tool that
        # exists, and a contract only off a name that is advertised.
        self._short_to_full: Dict[str, str] = {}
        self._schemas = self._advertise()
        self._contracts = self._build_contracts()

    @property
    def note(self) -> str:
        """One line about how this runtime came up, for the server's banner."""
        llm = 'llm configured' if self.has_llm else 'no llm (read_file.abbreviate withdrawn)'
        return llm + (', permission: ' + '+'.join(self.permission_patches) if self.permission_patches else '')

    async def _prepare(self) -> None:
        self.agent.prepare_runtime()
        await self.agent.prepare_tools()
        # After prepare_tools, which is what builds the tool manager the skill
        # toolset registers itself into: skills_list / skill_view / skill_manage
        # then come back from get_tools() like any other tool and are advertised
        # without this side knowing they are skills. Called unconditionally --
        # with no ``skills`` section there are no skills to find, and the same
        # call on the training host (MsAgentHarness._prepare_async) is what puts
        # the skill bodies in the system prompt. Skipping it here is what would
        # need justifying: the prompt would name three tools this runtime cannot
        # run.
        await self.agent.prepare_skills()

    def _safety_switches(self) -> Tuple[bool, bool]:
        """``(unrestricted_removal, allow_write_globs)`` as configured.

        Absent means off, which is what an unmodified ms-agent does with these
        keys anyway -- so a config that never mentions them keeps every refusal.

        Warns when ``dangerous_removal_paths`` is configured alongside
        ``unrestricted_removal``, because the patch makes that list moot and a
        silently ignored blacklist is the one outcome worth shouting about.
        """
        permission = getattr(self.agent.config, 'permission', None)
        rules = (getattr(permission, 'safety_rules', None) or {}) if permission is not None else {}
        unrestricted = bool(_cfg_get(rules, 'unrestricted_removal', False))
        globs = bool(_cfg_get(rules, 'allow_write_globs', False))
        if unrestricted and _cfg_get(rules, 'dangerous_removal_paths', None):
            _warn('unrestricted_removal bypasses the rm/rmdir path check entirely, so the '
                  'configured dangerous_removal_paths list will not be consulted')
        return unrestricted, globs

    @property
    def _tm(self):
        return self.agent.tool_manager

    def _bare_name(self, name: str) -> str:
        """The tool's own name, without the ``{server}---`` prefix.

        First separator, not the last: a tool id may itself contain one, which is
        why ms-agent splits it this way too (``_registered_tool_suffix``).
        """
        splitter = type(self._tm).TOOL_SPLITER
        return name.split(splitter, 1)[1] if splitter in name else name

    def tools(self) -> List[Dict[str, Any]]:
        """The advertised schemas, built once at startup."""
        return self._schemas

    def _advertise(self) -> List[Dict[str, Any]]:
        """Tool schemas, OpenAI-shaped, named as the model will be shown them.

        ``get_tools`` already returns one flat, name-sorted list; the names are
        namespaced as ``{server}---{tool}``, and what this deployment cannot run
        is cut from the parameters.

        The namespace is then dropped wherever it is unambiguous, and
        :meth:`call` puts it back. A 4B policy spends whole calls on that prefix:
        across three arms it wrote a bare ``shell_executor`` 7 times, each one
        refused with "unknown tool ... Did you mean
        'code_executor---shell_executor'?" -- a turn burnt on punctuation.
        Since the prefix carries nothing the model can act on (nothing here has
        two servers offering the same tool), it is not shown one.

        This is not the same as accepting a wrong name and fixing it up: the
        model is shown ``shell_executor`` and calls ``shell_executor``, so what
        it learns to emit is what the schema promised. A name that would collide
        keeps its prefix, in both directions, rather than becoming ambiguous.
        """
        schemas = [_to_openai(t) for t in self._loop.run(self._tm.get_tools()) if isinstance(t, dict)]
        schemas = [_without_args(t, _INTERNAL_ARGS) for t in schemas]
        if not self.has_llm:
            schemas = [
                _without_args(t, _LLM_BACKED_ARGS.get((t.get('function') or {}).get('name'), ())) for t in schemas
            ]
        bare_counts: Dict[str, int] = {}
        for schema in schemas:
            name = (schema.get('function') or {}).get('name')
            if name:
                bare = self._bare_name(str(name))
                bare_counts[bare] = bare_counts.get(bare, 0) + 1
        advertised = []
        for schema in schemas:
            fn = schema.get('function') or {}
            full = str(fn.get('name') or '')
            bare = self._bare_name(full)
            if full and bare != full and bare_counts.get(bare) == 1:
                schema = copy.deepcopy(schema)
                schema['function']['name'] = bare
                self._short_to_full[bare] = full
            advertised.append(schema)
        return advertised

    def _build_contracts(self) -> Dict[str, Tuple[Set[str], Optional[Set[str]]]]:
        """Per tool: the arguments advertised, and the ones the code will take.

        Both halves are needed because ms-agent lets them disagree, and every
        disagreement is a call the model was invited to make and cannot. The
        advertised half comes from :meth:`tools`, so it is the exact contract the
        prompt carries; the other from the signature of the method ``call_tool``
        will ``getattr`` and splat the arguments into. ``None`` means the method
        takes ``**kwargs`` or could not be introspected -- then nothing is
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
            for name, (declared, accepted) in contracts.items() if accepted is not None and declared - accepted
        }
        if drift:
            _warn('advertised arguments the implementation rejects '
                  '(dropped at dispatch, fix upstream): %s' % (drift, ))
        return contracts

    def _accepted_args(self, name: str) -> Optional[Set[str]]:
        """Keyword names the implementation behind ``name`` accepts, or None.

        ``name`` is an advertised one, so the index is looked up under the
        runtime's own spelling -- the tool index knows nothing about the short
        names this file invents.
        """
        full = self._short_to_full.get(name, name)
        try:
            tool_ins = self._tm._tool_index[full][0]
            sig = inspect.signature(getattr(tool_ins, self._bare_name(full)))
        except Exception:  # noqa -- an un-introspectable tool just gets no repairs
            return None
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return None
        return {n for n, p in sig.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

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
        # glob's own default for ``path`` is '' and it is advertised as optional,
        # but ms-agent's safety guard rejects an empty file path before dispatch,
        # so a model that spells the default out loud gets "Blocked by safety
        # policy: Empty file path" -- 66 times in 5793 calls. '.' is what ''
        # resolves to once inside the tool.
        if self._bare_name(name) == 'glob' and 'path' in args and not str(args.get('path') or '').strip():
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
        parts = [
            'Error: %s has no argument %s.' % (name, ', '.join(repr(a) for a in unknown)),
            'It accepts: %s.' % (', '.join(sorted(declared)) or '(none)'),
        ]
        for arg, tools in sorted(owners.items()):
            parts.append('%r belongs to %s.' % (arg, ' or '.join(sorted(tools))))
        parts.append("Re-issue the call with this tool's arguments, or call the tool "
                     'the arguments belong to.')
        return ' '.join(parts)

    def call(self, calls: List[Dict[str, Any]], timeout: Optional[float]) -> List[Dict[str, Any]]:
        """Dispatch a turn's tool calls, mirroring how ms-agent itself does it.

        A single call goes through ``single_call_tool`` and a batch through
        ``parallel_call_tool``, matching LLMAgent, so concurrency-sensitive tools
        behave in training exactly as they do in production. Each call is put
        through :meth:`_reconcile` first, and one that cannot run is answered from
        here without reaching ms-agent -- so a batch keeps its shape and result
        *i* still answers call *i*.

        Names arrive as advertised and are dispatched under ms-agent's own
        spelling (see :meth:`_advertise`). An unmapped name goes through as-is,
        which is both what a caller using the namespaced form wants and how a
        genuinely unknown tool reaches ms-agent's own "did you mean" reply.
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
                    prepared.append((i, {'tool_name': self._dispatch_name(name), 'arguments': c.get('arguments')}))
                    continue
            if not isinstance(args, dict):
                args = {}
            args, error = self._reconcile(name, args)
            if error:
                out[i] = {'observation': error, 'ok': False}
            else:
                prepared.append((i, {'tool_name': self._dispatch_name(name), 'arguments': args}))
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
                detail = self._failure_text(e, timeout)
                for i, _p in prepared:
                    out[i] = {'observation': detail, 'ok': False}
            else:
                for (i, _p), r in zip(prepared, list(results)):
                    out[i] = {'observation': _with_timeout_advice(_as_text(r)), 'ok': True}
        return [o if o is not None else {'observation': '', 'ok': False} for o in out]

    def _dispatch_name(self, name: Optional[str]) -> Optional[str]:
        """ms-agent's own spelling for a name taken from a tool call."""
        return self._short_to_full.get(name, name) if name else name

    @staticmethod
    def _failure_text(exc: BaseException, timeout: Optional[float]) -> str:
        """What the model reads when the whole turn's dispatch failed.

        A timeout is spelled out rather than reported as its exception name.
        ``concurrent.futures.TimeoutError`` carries no message at all, so the
        model used to read "Tool call failed. TimeoutError:" -- which says
        nothing about what to do differently. What it needs to know is that the
        call was abandoned rather than rejected, and that whatever it started may
        still be running: this cannot cancel a subprocess ms-agent has spawned.
        """
        if isinstance(exc, (FuturesTimeoutError, asyncio.TimeoutError)):
            return (f"Timed out: this turn's tool calls did not finish within {timeout}s and were "
                    f'abandoned. Whatever they started may still be running. {_TIMEOUT_ADVICE}')
        return f'Tool call failed. {type(exc).__name__}: {exc}'


def _with_timeout_advice(observation: str) -> str:
    if _MS_TIMEOUT_MARK in observation and 'run_in_background' not in observation:
        return observation + ' ' + _TIMEOUT_ADVICE
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


def _warn(message: str) -> None:
    sys.stderr.write('[runtime_msagent] WARNING %s\n' % (message, ))
    sys.stderr.flush()
