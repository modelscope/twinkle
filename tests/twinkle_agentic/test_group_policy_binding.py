import pytest

from twinkle_agentic.async_rl.rollout_resume import GroupPolicyBinding


def test_group_policy_binding_round_trip():
    binding = GroupPolicyBinding(
        state_key="run/group/17",
        policy_version=4,
        adapter_path="/policies/policy-4",
    )
    assert GroupPolicyBinding.from_dict(binding.to_dict()) == binding


@pytest.mark.parametrize(
    ("state_key", "policy_version", "adapter_path", "message"),
    [
        ("", 0, None, "state_key"),
        ("run/group/0", -1, None, "policy_version"),
        ("run/group/0", 0, "", "adapter_path"),
    ],
)
def test_group_policy_binding_rejects_invalid_identity(state_key, policy_version, adapter_path, message):
    with pytest.raises(ValueError, match=message):
        GroupPolicyBinding(state_key, policy_version, adapter_path)


def test_group_policy_binding_rejects_bool_version_from_snapshot():
    with pytest.raises(ValueError, match="policy_version"):
        GroupPolicyBinding.from_dict({"state_key": "run/group/0", "policy_version": True})
