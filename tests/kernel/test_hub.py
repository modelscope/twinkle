import pytest

from twinkle.kernel.core import HubRef, hub


def test_hub_with_version():
    ref = hub('kernels-community/activation:SiluAndMul', version=1)
    assert isinstance(ref, HubRef)
    assert ref.repo_id == 'kernels-community/activation'
    assert ref.layer_name == 'SiluAndMul'
    assert ref.version == 1
    assert ref.revision is None
    assert ref.backend is None
    assert ref.trust_remote_code is False


def test_hub_with_revision():
    ref = hub('org/repo:Layer', revision='main')
    assert ref.revision == 'main'
    assert ref.version is None


def test_hub_with_backend_and_trust():
    ref = hub('org/repo:Layer', version=2, backend='cuda', trust_remote_code=True)
    assert ref.backend == 'cuda'
    assert ref.trust_remote_code is True


def test_hub_rejects_both_revision_and_version():
    with pytest.raises(ValueError, match='Exactly one'):
        hub('org/repo:Layer', revision='main', version=1)


def test_hub_rejects_neither_revision_nor_version():
    with pytest.raises(ValueError, match='Exactly one'):
        hub('org/repo:Layer')


def test_hub_rejects_missing_colon():
    with pytest.raises(ValueError, match='repo_id:LayerName'):
        hub('org/repo', version=1)


def test_hub_handles_colon_in_repo_id():
    # rsplit takes only the last colon
    ref = hub('org:sub/repo:Layer', version=1)
    assert ref.repo_id == 'org:sub/repo'
    assert ref.layer_name == 'Layer'


def test_hubref_is_frozen():
    ref = hub('org/repo:Layer', version=1)
    with pytest.raises(Exception):
        ref.repo_id = 'other'