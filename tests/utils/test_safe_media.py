# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the media-safety guards in twinkle.utils.safe_media."""
import pytest

from twinkle.utils.safe_media import MEDIA_ALLOWED_DIRS_ENV, _assert_public_host, checked_media_path


class TestAssertPublicHost:

    def test_public_address_passes(self):
        _assert_public_host('8.8.8.8')

    def test_loopback_and_private_ranges_are_refused(self):
        for host in ('127.0.0.1', '10.0.0.1', '192.168.1.1', '172.16.0.1'):
            with pytest.raises(ValueError, match='non-public address'):
                _assert_public_host(host)

    def test_cloud_metadata_link_local_is_refused(self):
        # 169.254.169.254 is the cloud metadata endpoint: link-local range.
        with pytest.raises(ValueError, match='non-public address'):
            _assert_public_host('169.254.169.254')

    def test_unresolvable_host_is_refused(self):
        with pytest.raises(ValueError, match='cannot resolve host'):
            _assert_public_host('nonexistent-host-xyz.invalid')


class TestCheckedMediaPath:

    def test_unset_allowlist_leaves_paths_unrestricted(self, monkeypatch, tmp_path):
        monkeypatch.delenv(MEDIA_ALLOWED_DIRS_ENV, raising=False)
        assert checked_media_path(str(tmp_path / 'anywhere.png')) == str(tmp_path / 'anywhere.png')

    def test_path_inside_an_allowed_root_passes(self, monkeypatch, tmp_path):
        allowed = tmp_path / 'allowed'
        allowed.mkdir()
        target = allowed / 'sub' / 'img.png'
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, str(allowed))
        assert checked_media_path(str(target)) == str(target)

    def test_path_outside_the_allowlist_is_refused(self, monkeypatch, tmp_path):
        allowed = tmp_path / 'allowed'
        allowed.mkdir()
        outside = tmp_path / 'outside' / 'img.png'
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, str(allowed))
        with pytest.raises(ValueError, match='outside'):
            checked_media_path(str(outside))

    def test_dotdot_traversal_out_of_the_allowlist_is_refused(self, monkeypatch, tmp_path):
        allowed = tmp_path / 'allowed'
        allowed.mkdir()
        traversal = str(allowed / '..' / 'secret.png')
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, str(allowed))
        with pytest.raises(ValueError, match='outside'):
            checked_media_path(traversal)

    def test_symlink_escape_is_refused(self, monkeypatch, tmp_path):
        allowed = tmp_path / 'allowed'
        allowed.mkdir()
        secret = tmp_path / 'secret.png'
        secret.write_bytes(b'x')
        link = allowed / 'link.png'
        link.symlink_to(secret)
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, str(allowed))
        with pytest.raises(ValueError, match='outside'):
            checked_media_path(str(link))

    def test_prefix_similar_directory_is_not_confused(self, monkeypatch, tmp_path):
        # 'allowed_sub' shares a string prefix with 'allowed' but is a different
        # directory: commonpath must not treat it as inside.
        allowed = tmp_path / 'allowed'
        allowed.mkdir()
        sibling = tmp_path / 'allowed_sub'
        sibling.mkdir()
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, str(allowed))
        with pytest.raises(ValueError, match='outside'):
            checked_media_path(str(sibling / 'img.png'))

    def test_multiple_roots_are_all_honoured(self, monkeypatch, tmp_path):
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        first.mkdir()
        second.mkdir()
        monkeypatch.setenv(MEDIA_ALLOWED_DIRS_ENV, f'{first}{__import__("os").pathsep}{second}')
        assert checked_media_path(str(second / 'img.png')) == str(second / 'img.png')
