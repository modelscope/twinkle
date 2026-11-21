import pytest

from twinkle_agentic.rollout.ledger import TurnLedger


class _Template:
    max_length = 32


def test_turn_ledger_round_trip_keeps_exact_sampler_state():
    ledger = TurnLedger(_Template(), label="gate0")
    ledger._pif = {
        "input_ids": [10, 11, 12, 13],
        "labels": [-100, -100, 12, 13],
        "completion_mask": [0, 0, 1, 1],
        "response_positions": [1, 2],
        "messages": [{"role": "user", "content": "hello"}],
    }
    ledger._logprobs = [[(12, -0.1)], [(13, -0.2)]]
    ledger._sampling_masks = {"blocks": [1, 2]}
    ledger._turns = 2

    state = ledger.state_dict()
    restored = TurnLedger(_Template(), label="gate0")
    restored.load_state_dict(state)

    assert restored.input_feature == ledger.input_feature
    assert restored.logprobs == ledger.logprobs
    assert restored.turns == 2
    restored.input_feature["input_ids"].append(99)
    assert ledger.input_feature["input_ids"] == [10, 11, 12, 13]


def test_turn_ledger_refuses_an_incomplete_checkpoint():
    ledger = TurnLedger(_Template())
    with pytest.raises(ValueError, match="missing input_feature"):
        ledger.load_state_dict({"format": "twinkle.turn-ledger.v1", "input_feature": {}, "logprobs": [], "turns": 0})
