# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import twinkle
from twinkle.utils.loader import Plugin, construct_class
from twinkle.utils.unsafe import trust_remote_code

twinkle.initialize(mode='local')


class BasePlugin:
    """Base class for testing plugins."""

    def __init__(self, name: str = 'default'):
        self.name = name


class SamplePlugin(BasePlugin):
    """Sample plugin class for testing."""
    pass


class TestPluginLoad:
    """Test Plugin.load_plugin against the current loader contract.

    PLAN (case | input | expected | failure it kills):
      nonexistent_local_path | bare id that is no file/folder | FileNotFoundError 'neither a file
        nor a folder' | silent success / wrong error class on a typo'd plugin path
      unknown_scheme | 'foo://x' | FileNotFoundError (only hf/ms route to hub) | unrecognized
        schemes silently downloading instead of failing as a missing local path
      missing_init | folder without __init__.py | FileNotFoundError 'has no __init__.py' |
        importing an empty folder as a plugin
      no_subclass | module defining only unrelated classes | ValueError 'Cannot find any subclass'
        | accepting a plugin that does not extend the kernel base
      reexport_not_own | module importing a subclass but defining none | ValueError | a merely
        re-exported sibling subclass mistaken for the plugin's own (__module__ gate)
      ms_hub / hf_hub | mocked download -> real plugin file | the defined subclass returned |
        hub routing broken (ms:// vs hf://) or the download gate bypassed
      sys_path_and_identity | two distinct plugin folders | both stay importable, distinct classes,
        same path twice -> identical class | the old fixed '__init__' module name handing plugin
        #2 back plugin #1's classes; cache misses re-importing
    """

    def test_load_plugin_nonexistent_local_path(self):
        with pytest.raises(FileNotFoundError, match='neither a file nor a folder'):
            Plugin.load_plugin('invalid_id', BasePlugin)

    def test_load_plugin_unknown_scheme(self):
        # Only hf:// and ms:// route to the hub; anything else is a local path, so an
        # unrecognized scheme fails loudly as a missing path rather than silently downloading.
        with pytest.raises(FileNotFoundError, match='neither a file nor a folder'):
            Plugin.load_plugin('foo://test/plugin', BasePlugin)

    def test_load_plugin_safe_mode(self):
        """Test loading plugin when trust_remote_code is False."""
        with patch('twinkle.hub.hub.HubOperation.download_model', return_value='/tmp/fake'):
            with patch('twinkle.utils.loader.trust_remote_code', return_value=False):
                with pytest.raises(ValueError, match='Twinkle does not support plugin in safe mode'):
                    Plugin.load_plugin('ms://test/plugin', BasePlugin)

    def test_load_plugin_missing_init_file(self):
        """Test loading plugin when __init__.py is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with pytest.raises(FileNotFoundError, match='has no __init__.py'):
                        Plugin.load_plugin('ms://test/plugin', BasePlugin)

    def _write_plugin(self, tmpdir, body):
        init_file = Path(tmpdir) / '__init__.py'
        init_file.write_text(body)
        return init_file

    _PLUGIN_BODY = ('from tests.pluger.test_loader import BasePlugin\n'
                    'class TestPlugin(BasePlugin):\n'
                    '    pass\n')

    def test_load_plugin_no_subclass(self):
        """A module with only unrelated classes must be refused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_plugin(tmpdir, 'class OtherClass:\n    pass\n')
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with pytest.raises(ValueError, match='Cannot find any subclass'):
                        Plugin.load_plugin('ms://test/plugin', BasePlugin)

    def test_load_plugin_reexported_subclass_not_own(self):
        """Importing a subclass without defining one must not count (__module__ gate)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_plugin(tmpdir, 'from tests.pluger.test_loader import SamplePlugin\n')
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with pytest.raises(ValueError, match='Cannot find any subclass'):
                        Plugin.load_plugin('ms://test/plugin', BasePlugin)

    def test_load_plugin_ms_hub(self):
        """Test loading plugin from ModelScope hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_plugin(tmpdir, self._PLUGIN_BODY)
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir) as dl:
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    plugin_cls = Plugin.load_plugin('ms://test/plugin', BasePlugin)
                    assert plugin_cls.__name__ == 'TestPlugin'
                    assert issubclass(plugin_cls, BasePlugin)
                    dl.assert_called_once()

    def test_load_plugin_hf_hub(self):
        """Test loading plugin from HuggingFace hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_plugin(tmpdir, self._PLUGIN_BODY)
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir) as dl:
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    plugin_cls = Plugin.load_plugin('hf://test/plugin', BasePlugin)
                    assert plugin_cls.__name__ == 'TestPlugin'
                    assert issubclass(plugin_cls, BasePlugin)
                    dl.assert_called_once()

    def test_load_plugin_sys_path_and_identity(self):
        """The plugin folder stays importable; distinct plugins never collide; reload is cached."""
        with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
            self._write_plugin(tmpdir_a, self._PLUGIN_BODY)
            self._write_plugin(tmpdir_b, self._PLUGIN_BODY)
            with patch('twinkle.hub.hub.HubOperation.download_model', side_effect=[tmpdir_a, tmpdir_b, tmpdir_a]):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    first = Plugin.load_plugin('ms://test/plugin', BasePlugin)
                    second = Plugin.load_plugin('ms://test/plugin', BasePlugin)
                    again = Plugin.load_plugin('ms://test/plugin', BasePlugin)
            # The old fixed '__init__' module name returned plugin A's class for plugin B.
            assert first is not second
            assert first is again  # same resolved path -> cached, not re-imported
            # Siblings of a folder plugin must stay importable for the process lifetime.
            assert str(Path(tmpdir_a).resolve()) in sys.path
            assert str(Path(tmpdir_b).resolve()) in sys.path


class TestConstructClass:
    """Test construct_class functionality."""

    def test_construct_class_with_instance(self):
        """Test construct_class when input is already an instance."""
        instance = SamplePlugin('test')
        result = construct_class(instance, BasePlugin, [])
        assert result is instance

    def test_construct_class_with_class_type(self):
        """Test construct_class when input is a class type."""
        result = construct_class(SamplePlugin, BasePlugin, [], name='test')
        assert isinstance(result, SamplePlugin)
        assert result.name == 'test'

    def test_construct_class_with_string_name(self):
        """Test construct_class when input is a string class name."""
        # Import the current test module to access SamplePlugin
        import sys
        current_module = sys.modules[__name__]
        result = construct_class('SamplePlugin', BasePlugin, [current_module], name='test')
        assert isinstance(result, SamplePlugin)
        assert result.name == 'test'

    def test_construct_class_with_plugin_id(self):
        """Test construct_class when input is a plugin ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / '__init__.py'
            init_file.write_text('from tests.pluger.test_loader import BasePlugin\n'
                                 'class TestPlugin(BasePlugin):\n'
                                 '    pass\n')
            with patch('twinkle.hub.hub.HubOperation.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    result = construct_class('ms://test/plugin', BasePlugin, [], name='test')
                    assert isinstance(result, BasePlugin)
                    assert result.name == 'test'

    def test_construct_class_with_invalid_input(self):
        """Test construct_class with invalid input type."""
        result = construct_class(123, BasePlugin, [])
        assert result == 123
