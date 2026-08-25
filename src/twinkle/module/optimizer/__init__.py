# Copyright (c) ModelScope Contributors. All rights reserved.
"""The optimizers `set_optimizer` can resolve by name, on top of the ones in `torch.optim`.

Each family is a subpackage and re-exported here, because that is the name
`twinkle.model.transformers.transformers` hands to `construct_class` -- a name looked up on this
module, not on the subpackages. Without this file the directory is an implicit namespace package with
no attributes, so every optimizer here is unreachable by name and the module-level import of
`GaLoreConfig` fails outright.
"""
from .galore import (GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit, GaLoreConfig, GaLoreProjector,
                     create_galore_param_groups)
from .muon import MaxLogitsTracker, MuonClip, MuonConfig, create_muon_param_groups
