"""defect58 part② EngineCore reaper unit tests (CPU logic; GPU ray-SIGTERM e2e is separate).

PLAN (case | input | expected | failure it kills):
  record_actor_writes   | ray actor ctx + proc-manager engine | PID file with pids   | record blind to real PIDs (false-neg reaper)
  record_nonactor_noop  | ray init, no actor_id               | no file              | spurious local-engine record
  record_version_guard  | engine lacks engine_core link       | no file, no raise    | crash on older vLLM layout
  reap_matching_kills   | PID file -> real EngineCore proc    | SIGTERM, proc dies   | reaper blind to real orphan (false-neg)
  reap_pid_reuse_spares | PID file -> plain sleep proc        | NOT killed           | mis-reap unrelated process (false-pos)
  reap_dead_pid_no_raise| PID file -> dead pid                | skipped, no raise    | crash on already-dead orphan
  reap_job_isolation    | two job files, reap job A           | only A's pids killed | cross-job mis-reap
  reap_no_file_noop     | nonexistent job                     | [] no raise          | crash on no evidence
  reverse_verify_guard  | EngineCore proc / sleep proc / self | True / False / False | insensitive identity guard (lowercase bug)

Reap cases drive REAL subprocesses (setproctitle 'VLLM::EngineCore'); no mock of the kill path.
Record cases isolate the ray runtime context (unit-level); the real ray API + actor lifecycle are
exercised by the GPU ray-SIGTERM end-to-end chain, not substituted here.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from twinkle.sampler.vllm_sampler import engine_core_reaper as rpr


def _spawn(title):
    """Spawn a real subprocess whose proc title is `title` (faithful EngineCore proxy via setproctitle)."""
    code = ('import setproctitle, time, sys; '
            f'setproctitle.setproctitle({title!r}); '
            'sys.stdout.write("ready\\n"); sys.stdout.flush(); time.sleep(60)')
    p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'ready'
    return p


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid


class _FakeEngine:
    """Duck-typed engine exposing the vLLM 0.23 engine_core.resources.engine_manager.processes path."""
    def __init__(self, pids):
        self.engine_core = mock.Mock()
        self.engine_core.resources.engine_manager.processes = [_FakeProc(p) for p in pids]


class EngineCoreReaperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = rpr.ENGINE_CORE_PID_DIR
        rpr.ENGINE_CORE_PID_DIR = self._tmp.name
        self._procs = []

    def tearDown(self):
        for p in self._procs:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        rpr.ENGINE_CORE_PID_DIR = self._orig_dir
        self._tmp.cleanup()

    def _write_rec(self, job, pids, actor='actor0'):
        path = os.path.join(rpr.ENGINE_CORE_PID_DIR, f'{job}__{actor}.json')
        with open(path, 'w') as f:
            json.dump({'job_id': job, 'actor_name': actor, 'actor_pid': os.getpid(),
                       'engine_core_pids': pids, 'cuda_visible_devices': '0,1'}, f)
        return path

    # ---- record (ray context isolated at unit level) ----
    def _record_with_ctx(self, engine, actor_id):
        ctx = mock.Mock()
        ctx.get_actor_id.return_value = actor_id
        ctx.get_job_id.return_value = 'jobTEST'
        ctx.get_actor_name.return_value = 'sampler0'
        with mock.patch('ray.is_initialized', return_value=True), \
             mock.patch('ray.get_runtime_context', return_value=ctx):
            rpr.record_engine_core_pids(engine)

    def test_record_actor_writes(self):
        self._record_with_ctx(_FakeEngine([111, 222]), actor_id='actor123')
        path = os.path.join(rpr.ENGINE_CORE_PID_DIR, 'jobTEST__sampler0.json')
        self.assertTrue(os.path.exists(path))
        rec = json.load(open(path))
        self.assertEqual(rec['engine_core_pids'], [111, 222])

    def test_record_nonactor_noop(self):
        self._record_with_ctx(_FakeEngine([111]), actor_id=None)  # not an actor
        self.assertEqual(os.listdir(rpr.ENGINE_CORE_PID_DIR), [])

    def test_record_version_guard(self):
        self._record_with_ctx(object(), actor_id='actor123')  # no engine_core link
        self.assertEqual(os.listdir(rpr.ENGINE_CORE_PID_DIR), [])  # no raise, no file

    # ---- reap (real subprocesses) ----
    def test_reap_matching_kills(self):
        p = _spawn('VLLM::EngineCore'); self._procs.append(p)
        self._write_rec('jobA', [p.pid])
        signalled = rpr.reap_engine_core_orphans('jobA')
        self.assertIn(p.pid, signalled)
        p.wait(timeout=10)
        self.assertFalse(_alive(p.pid))  # really reaped

    def test_reap_pid_reuse_spares(self):
        p = _spawn('plain_sleep_worker'); self._procs.append(p)  # NOT an EngineCore
        self._write_rec('jobB', [p.pid])
        signalled = rpr.reap_engine_core_orphans('jobB')
        self.assertEqual(signalled, [])  # identity guard spares it
        self.assertTrue(_alive(p.pid))

    def test_reap_dead_pid_no_raise(self):
        p = _spawn('VLLM::EngineCore'); self._procs.append(p)
        pid = p.pid
        p.kill(); p.wait(timeout=5)
        self._write_rec('jobC', [pid])
        signalled = rpr.reap_engine_core_orphans('jobC')  # must not raise
        self.assertEqual(signalled, [])

    def test_reap_job_isolation(self):
        pa = _spawn('VLLM::EngineCore'); self._procs.append(pa)
        pb = _spawn('VLLM::EngineCore'); self._procs.append(pb)
        self._write_rec('jobA', [pa.pid], actor='a0')
        self._write_rec('jobB', [pb.pid], actor='b0')
        signalled = rpr.reap_engine_core_orphans('jobA')
        self.assertEqual(signalled, [pa.pid])  # only job A
        pa.wait(timeout=10)
        self.assertTrue(_alive(pb.pid))  # job B untouched

    def test_reap_no_file_noop(self):
        self.assertEqual(rpr.reap_engine_core_orphans('nonexistent_job'), [])

    # ---- reverse-verify the identity guard is load-bearing ----
    def test_reverse_verify_guard(self):
        p = _spawn('VLLM::EngineCore'); self._procs.append(p)
        s = _spawn('plain_sleep_worker'); self._procs.append(s)
        self.assertTrue(rpr._looks_like_engine_core(p.pid))   # EngineCore -> True
        self.assertFalse(rpr._looks_like_engine_core(s.pid))  # unrelated -> False
        self.assertFalse(rpr._looks_like_engine_core(os.getpid()))  # self (imports module) -> False
        # reverse: a guard that matched lowercase 'engine_core' would mis-flag self -> the bug fixed
        self.assertNotIn('EngineCore', open(f'/proc/{os.getpid()}/cmdline', 'rb').read().decode('utf-8', 'replace'))


if __name__ == '__main__':
    unittest.main()