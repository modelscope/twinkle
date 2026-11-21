# Copyright (c) ModelScope Contributors. All rights reserved.
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
