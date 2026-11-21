# Copyright (c) ModelScope Contributors. All rights reserved.
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from twinkle.hub import HubOperation
from twinkle.hub.hub import HFHub, MSHub


def test_modelscope_upload_uses_the_sdk_repository_creation(monkeypatch, tmp_path):
    upload = Mock(return_value=True)
    monkeypatch.setattr('modelscope.push_to_hub', upload)
    create = Mock()
    monkeypatch.setattr(MSHub, 'create_model_repo', create)

    HubOperation.push_to_hub('owner/checkpoint', tmp_path, token='test-token', private=True)

    create.assert_not_called()
    upload.assert_called_once_with(
        'owner/checkpoint', tmp_path, token='test-token', private=True, commit_message='Upload folder using api',
        ignore_file_pattern=None, revision='master', tag=None)


def test_huggingface_checkpoint_reuses_upload_and_returns_commit(monkeypatch, tmp_path: Path):
    monkeypatch.delenv('TWINKLE_FORBID_HF', raising=False)
    upload = Mock(return_value=SimpleNamespace(oid='hf-commit'))
    monkeypatch.setattr(HFHub, 'push_to_hub', upload)
    report = Mock()

    assert HubOperation.upload_checkpoint('hf://owner/checkpoint', str(tmp_path), 'test-token', report) == 'hf-commit'

    assert upload.call_args.args[:2] == ('owner/checkpoint', str(tmp_path))
    report.assert_not_called()


def test_modelscope_checkpoint_reads_commit_id_from_real_response_shape(monkeypatch, tmp_path):
    pytest.importorskip('modelscope_hub')
    api = Mock()
    api.upload_folder.return_value = {'commit_id': '070e2c1ac96cc6e7684356a7d4e1677f486cf04c'}
    monkeypatch.setattr('modelscope_hub.HubApi', Mock(return_value=api))
    monkeypatch.setattr(MSHub, '_ensure_configuration', Mock())
    report = Mock()

    revision = MSHub.upload_checkpoint('owner/checkpoint', str(tmp_path), 'test-token', report)

    assert revision == '070e2c1ac96cc6e7684356a7d4e1677f486cf04c'
    api.create_repo.assert_called_once_with('owner/checkpoint', 'model', visibility='private')
    assert api.upload_folder.call_args.kwargs['use_cache'] is False
    assert callable(api.upload_folder.call_args.kwargs['progress_callback'])
