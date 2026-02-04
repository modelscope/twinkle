# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from torch import nn

from twinkle.model.moe.expert_parallel import ExpertParallelConfig, _merge_config, shard_experts


class _DummyMesh:
    ep_world_size = 2
    ep_rank = 0


class _DummyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.gate = nn.Linear(4, 2)
        self.num_experts = 2


class TestExpertParallelEfsdpGuard(unittest.TestCase):

    def test_merge_config_accepts_efsdp_bool(self):
        cfg = _merge_config({"efsdp": True})
        self.assertTrue(cfg.efsdp)

    def test_merge_config_accepts_efsdp_dict(self):
        cfg = _merge_config({"efsdp": {"enabled": True, "shard_dim": 1, "mesh_dim": "dp"}})
        self.assertTrue(cfg.efsdp)
        self.assertEqual(cfg.efsdp_shard_dim, 1)
        self.assertEqual(cfg.efsdp_mesh_dim, "dp")

    def test_merge_config_rejects_unknown_efsdp_key(self):
        with self.assertRaisesRegex(ValueError, "Unknown efsdp config"):
            _merge_config({"efsdp": {"foo": True}})

    def test_modulelist_experts_raise_when_efsdp_enabled(self):
        block = _DummyMoeBlock()
        cfg = ExpertParallelConfig(efsdp=True)
        with self.assertRaisesRegex(NotImplementedError, "ModuleList experts"):
            shard_experts(block, _DummyMesh(), cfg)


if __name__ == "__main__":
    unittest.main()
