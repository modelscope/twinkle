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
