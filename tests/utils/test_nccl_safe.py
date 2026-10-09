from unittest.mock import patch

import pytest

from twinkle.utils.nccl_safe import nccl_safe_megatron


def test_nccl_failure_preserves_type_and_adds_rank_context():

    @nccl_safe_megatron
    def fail(_self):
        raise ValueError('bad shape')

    with patch('twinkle.utils.nccl_safe._global_rank', return_value=3):
        with pytest.raises(ValueError) as caught:
            fail(object())

    assert 'global_rank=3' in ''.join(getattr(caught.value, '__notes__', caught.value.args))
