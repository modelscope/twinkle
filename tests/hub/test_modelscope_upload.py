# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import Mock

from twinkle.hub import HubOperation
from twinkle.hub.hub import MSHub


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
