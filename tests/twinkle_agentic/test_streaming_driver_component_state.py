import copy
from concurrent.futures import Future

from twinkle_agentic.async_rl import StreamingDriver
from twinkle_agentic.rollout.multi_turn import RolloutStatus


class _Rollout:
    def __init__(self):
        self.calls = []
        self.futures = []

    def step(self, trajectory, **_kwargs):
        self.calls.append(copy.deepcopy(trajectory))
        future = Future()
        self.futures.append(future)
        return future

    def stop(self, *_args, **_kwargs):
        return None


def _finish(driver, rollout):
    for future in rollout.futures:
        if not future.done():
            future.set_result(([], RolloutStatus(done=True)))
    driver.close()


def test_component_state_persists_raw_query_and_replays_it():
    query = {
        "messages": [
            {"role": "system", "content": "use the calculator"},
            {"role": "user", "content": "2 + 3"},
        ],
        "tools": [{"name": "calculator"}],
    }
    source_rollout = _Rollout()
    source = StreamingDriver([query], source_rollout, batch_size=1, num_generations=1)
    source._admit()

    state = source.state_dict()

    assert state["format"] == "twinkle.streaming-driver.v1"
    assert state["queries"] == [
        {
            "key": "0",
            "row": query,
        }
    ]
    assert "traj" not in state["queries"][0]
    _finish(source, source_rollout)

    restored_rollout = _Rollout()
    restored = StreamingDriver([], restored_rollout, batch_size=1, num_generations=1)
    restored.load_state_dict(state)

    assert restored._groups["0"].row == query
    assert restored_rollout.calls[0]["messages"] == query["messages"]
    assert restored_rollout.calls[0]["tools"] == query["tools"]
    _finish(restored, restored_rollout)


def test_policy_advance_does_not_delete_an_inflight_tool_query():
    class _Manager:
        def __init__(self):
            self.current = 0

        def version(self):
            return self.current

    manager = _Manager()
    rollout = _Rollout()
    query = {
        "messages": [{"role": "user", "content": "call the search tool"}],
        "tools": [{"name": "search"}],
    }
    driver = StreamingDriver(
        [query],
        rollout,
        batch_size=1,
        num_generations=1,
        checkpoint_manager=manager,
    )
    driver._version = 0
    driver._admit()

    manager.current = 1
    driver._version = 1
    state = driver.state_dict()

    assert driver._groups["0"].row == query
    assert state["queries"][0]["row"] == query
    _finish(driver, rollout)


def test_component_snapshot_does_not_mutate_scheduling_state():
    rollout = _Rollout()
    driver = StreamingDriver(
        [{"messages": [{"role": "user", "content": "one"}]}],
        rollout,
        batch_size=1,
        num_generations=1,
    )
    driver._admit()
    before = (set(driver._groups), driver._live, driver._inflight, dict(driver._counters))
    state = driver.state_dict()

    assert driver._groups.keys() == before[0]
    assert (driver._live, driver._inflight, driver._counters) == before[1:]
    assert state["queries"][0]["row"] == {"messages": [{"role": "user", "content": "one"}]}
    _finish(driver, rollout)


def _query(text):
    return {"messages": [{"role": "user", "content": text}]}


class _ResumeLoader:
    def __init__(self, epochs, batch_size=2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch = 0
        self.offset = 0
        self.set_epoch_calls = []

    def get_state(self):
        return {"version": 2, "epoch": self.epoch, "offset": self.offset}

    def load_state(self, state):
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        length = len(self.epochs[self.epoch])
        if self.offset >= length:
            self.epoch += self.offset // length
            self.offset %= length

    def set_epoch(self, epoch):
        self.set_epoch_calls.append(epoch)
        if epoch != self.epoch:
            self.offset = 0
        self.epoch = epoch

    def __iter__(self):
        rows = self.epochs[self.epoch]
        while self.offset < len(rows):
            batch = rows[self.offset:self.offset + self.batch_size]
            self.offset += len(batch)
            yield batch
        self.epoch += 1
        self.offset = 0


def _contents(rows):
    return [row["messages"][0]["content"] for row in rows]


def test_component_state_replays_partial_batch_once_across_snapshots():
    rows = [_query(f"q{index}") for index in range(4)]
    source_rollout = _Rollout()
    source = StreamingDriver(_ResumeLoader([rows]), source_rollout, batch_size=1, num_generations=1)
    source._admit()

    state = source.state_dict()
    assert state["loader_state"]["offset"] == 2
    assert state["pending_rows"] == [rows[1]]

    restored_rollout = _Rollout()
    restored = StreamingDriver(_ResumeLoader([rows]), restored_rollout, batch_size=1, num_generations=1)
    restored.load_state_dict(state)
    consecutive_state = restored.state_dict()
    assert consecutive_state["pending_rows"] == [rows[1]]

    second_rollout = _Rollout()
    second = StreamingDriver(_ResumeLoader([rows]), second_rollout, batch_size=1, num_generations=1)
    second.load_state_dict(consecutive_state)
    assert _contents(second._iter_rows()) == ["q1", "q2", "q3"]

    _finish(source, source_rollout)
    _finish(restored, restored_rollout)
    _finish(second, second_rollout)


def test_epoch_end_replays_pending_before_the_restored_epoch():
    epochs = [[_query("old-0"), _query("old-1")], [_query("new-0"), _query("new-1")]]
    source_rollout = _Rollout()
    source = StreamingDriver(
        _ResumeLoader(epochs), source_rollout, batch_size=1, num_generations=1, num_epochs=2)
    source._admit()
    state = source.state_dict()

    restored_loader = _ResumeLoader(epochs)
    restored_rollout = _Rollout()
    restored = StreamingDriver(restored_loader, restored_rollout, batch_size=1, num_generations=1, num_epochs=2)
    restored.load_state_dict(state)

    assert (restored_loader.epoch, restored_loader.offset) == (1, 0)
    assert _contents(restored._iter_rows()) == ["old-1", "new-0", "new-1"]
    assert restored_loader.set_epoch_calls == []
    _finish(source, source_rollout)
    _finish(restored, restored_rollout)


def test_v1_component_state_resumes_nonzero_loader_epoch():
    loader = _ResumeLoader([[_query("old")], [_query("new-0"), _query("new-1")]])
    driver = StreamingDriver(loader, _Rollout(), batch_size=1, num_generations=1, num_epochs=2)
    driver.load_state_dict({
        "format": "twinkle.streaming-driver.v1",
        "handed": 1,
        "prompts_drawn": 1,
        "policy_version": 0,
        "loader_state": {"version": 2, "epoch": 1, "offset": 1},
        "queries": [],
    })

    assert _contents(driver._iter_rows()) == ["new-1"]
    assert loader.set_epoch_calls == []
    driver.close()


def test_plain_iterable_resume_does_not_duplicate_pending_rows():
    batches = [[_query("q0"), _query("q1")], [_query("q2")]]
    source_rollout = _Rollout()
    source = StreamingDriver(batches, source_rollout, batch_size=1, num_generations=1)
    source._admit()
    state = source.state_dict()

    restored_rollout = _Rollout()
    restored = StreamingDriver(batches, restored_rollout, batch_size=1, num_generations=1)
    restored.load_state_dict(state)

    assert _contents(restored._iter_rows()) == ["q1", "q2"]
    _finish(source, source_rollout)
    _finish(restored, restored_rollout)


def test_completed_max_steps_restore_submits_no_rollout_and_ends_cleanly():
    rollout = _Rollout()
    driver = StreamingDriver([], rollout, batch_size=1, num_generations=1, max_steps=1)
    driver.load_state_dict({
        "format": "twinkle.streaming-driver.v1",
        "handed": 1,
        "prompts_drawn": 1,
        "policy_version": 0,
        "loader_state": None,
        "queries": [{"key": "0", "row": _query("done")}],
    })

    assert rollout.calls == []
    assert driver._next_batch() is None
    driver.close()
