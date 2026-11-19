"""Ray-independent reaping of vLLM EngineCore subprocesses (defect58 part②).

The vLLM v1 ``AsyncLLM`` spawns its EngineCore as a separate OS process via
``CoreEngineProcManager`` (vllm/v1/engine/utils.py). That process is NOT inside the ray actor
lifecycle: when the raylet dies first, ``cluster.sampler.shutdown()`` is a ray RPC that raises
``ActorDiedError`` and never runs ``VLLMEngine.shutdown()`` (which lives inside the actor), so the
EngineCore is reparented to init and holds the whole GPU's memory indefinitely. vLLM's own
``weakref.finalize`` also runs inside the actor and is skipped on SIGKILL. ``ray stop --force``
cannot reap it either (its process name contains no 'ray').

Fix: the actor records its EngineCore PIDs to a shared file at creation (ray healthy); the driver,
which captures the ray job id at setup, reaps those PIDs directly at teardown -- bypassing the dead
actor. The shared anchor is the ray job id (both sides derive it from the runtime context while ray
is healthy) plus a fixed directory, so no env var (raylet env inheritance is unreliable) and no
constructor plumbing through the sampler/builder chain is needed.

Safety: this is a best-effort safety net, never a hard dependency. Recording is gated on running as
a ray actor; reaping identity-verifies each PID (cmdline must still look like an EngineCore) before
SIGTERM so a reused PID is not killed. Every step isolates exceptions.
"""
import glob
import json
import os
import signal
import tempfile

import logging

logger = logging.getLogger(__name__)

#: Fixed shared directory both the actor (writer) and driver (reader) derive without env/plumbing.
ENGINE_CORE_PID_DIR = os.path.join(tempfile.gettempdir(), 'twinkle_engine_core_pids')


def _engine_core_pids(engine):
    """Best-effort read of the EngineCore subprocess PIDs from a vLLM AsyncLLM (version-guarded).

    vLLM 0.23 path: engine.engine_core (AsyncMPClient) -> .resources (BackgroundResources)
    -> .engine_manager (CoreEngineProcManager) -> .processes[*].pid. Older/other layouts differ;
    a missing link yields [] (the reaper is a safety net, not a hard dependency).
    """
    pids = []
    try:
        core = getattr(engine, 'engine_core', None)
        resources = getattr(core, 'resources', None)
        manager = getattr(resources, 'engine_manager', None)
        processes = getattr(manager, 'processes', None) or []
        for proc in processes:
            pid = getattr(proc, 'pid', None)
            if pid:
                pids.append(int(pid))
    except Exception as e:  # noqa: BLE001 - safety net must never break engine creation
        logger.warning('engine_core_reaper: could not read EngineCore PIDs: %r', e)
    return pids


def record_engine_core_pids(engine):
    """Actor side: record this engine's EngineCore PIDs keyed by ray job id + actor name.

    No-op unless running as a ray actor (local in-process engines die with the driver, no orphan).
    Called right after engine creation while ray is healthy. Never raises.
    """
    try:
        import ray
        if not ray.is_initialized():
            return
        ctx = ray.get_runtime_context()
        actor_id = ctx.get_actor_id()
        if not actor_id:  # not an actor (e.g. driver-side local engine)
            return
        pids = _engine_core_pids(engine)
        if not pids:
            return
        job_id = str(ctx.get_job_id())
        actor_name = ctx.get_actor_name() or str(actor_id)
        os.makedirs(ENGINE_CORE_PID_DIR, exist_ok=True)
        path = os.path.join(ENGINE_CORE_PID_DIR, f'{job_id}__{actor_name}.json')
        rec = {
            'job_id': job_id,
            'actor_name': actor_name,
            'actor_pid': os.getpid(),
            'engine_core_pids': pids,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rec, f)
        logger.info('engine_core_reaper: recorded EngineCore PIDs %s -> %s', pids, path)
    except Exception as e:  # noqa: BLE001
        logger.warning('engine_core_reaper: record failed (non-fatal): %r', e)


def _proc_cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().replace(b'\x00', b' ').decode('utf-8', 'replace')
    except OSError:
        return ''


def _looks_like_engine_core(pid):
    """Identity guard against PID reuse: the live process must still look like a vLLM EngineCore.

    Match the vLLM proc title ``VLLM::EngineCore`` (camelCase ``EngineCore``), NOT the generic
    lowercase substring ``engine_core`` -- the latter appears in any cmdline that merely imports
    this module (the driver, a test invocation) and would mis-reap unrelated processes.
    """
    cmdline = _proc_cmdline(pid)
    if not cmdline:
        return False
    return ('EngineCore' in cmdline) or ('VLLM::' in cmdline and 'Engine' in cmdline)


def reap_engine_core_orphans(job_id):
    """Driver side: reap this job's recorded EngineCore PIDs directly, bypassing the (dead) actor.

    Called in the teardown finally after sampler.shutdown() (which raises ActorDiedError when the
    raylet died). Identity-verifies each PID before SIGTERM; isolates per-PID and per-file errors so
    one failure cannot block the rest. Never raises. Returns the list of PIDs signalled.
    """
    signalled = []
    if not job_id:
        return signalled
    try:
        paths = glob.glob(os.path.join(ENGINE_CORE_PID_DIR, f'{job_id}__*.json'))
    except Exception:  # noqa: BLE001
        return signalled
    for path in paths:
        try:
            with open(path, encoding='utf-8') as f:
                rec = json.load(f)
            for pid in rec.get('engine_core_pids', []):
                try:
                    pid = int(pid)
                    if not _looks_like_engine_core(pid):
                        continue  # gone, or PID reused by an unrelated process -- do not kill
                    os.kill(pid, signal.SIGTERM)
                    signalled.append(pid)
                    logger.info('engine_core_reaper: reaped orphan EngineCore pid=%d (job=%s)', pid, job_id)
                except ProcessLookupError:
                    pass  # already dead
                except Exception as e:  # noqa: BLE001
                    logger.warning('engine_core_reaper: kill pid=%s failed (non-fatal): %r', pid, e)
        except Exception as e:  # noqa: BLE001
            logger.warning('engine_core_reaper: read %s failed (non-fatal): %r', path, e)
        finally:
            try:
                os.replace(path, path + '.reaped')  # mark consumed; keep for post-mortem (no rm)
            except OSError:
                pass
    return signalled