# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import twinkle
from twinkle.utils.loader import Plugin, construct_class
from twinkle.utils.unsafe import trust_remote_code

twinkle.initialize(mode='local')


class BasePlugin:
    """Base class for testing plugins."""
    def __init__(self, name: str = "default"):
        self.name = name


class SamplePlugin(BasePlugin):
    """Sample plugin class for testing."""
    pass


class TestPluginLoad:
    """Test Plugin.load_plugin functionality."""
    
    def test_load_plugin_invalid_id(self):
        """Test loading plugin with invalid ID format."""
        with pytest.raises(ValueError, match="Unknown plugin id"):
            Plugin.load_plugin("invalid_id", BasePlugin)
    
    def test_load_plugin_safe_mode(self):
        """Test loading plugin when trust_remote_code is False."""
        with patch('twinkle.utils.loader.trust_remote_code', return_value=False):
            with pytest.raises(ValueError, match="Twinkle does not support plugin in safe mode"):
                Plugin.load_plugin("ms://test/plugin", BasePlugin)
    
    def test_load_plugin_missing_init_file(self):
        """Test loading plugin when __init__.py is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('twinkle.utils.loader.MSHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with pytest.raises(AssertionError, match="does not exist"):
                        Plugin.load_plugin("ms://test/plugin", BasePlugin)
    
    def test_load_plugin_no_subclass(self):
        """Test loading plugin when no subclass of base class is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("class OtherClass:\n    pass\n")
            
            # Create a class that doesn't inherit from BasePlugin
            other_class = type('OtherClass', (), {})
            other_class.__module__ = str(init_file)
            
            with patch('twinkle.utils.loader.MSHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with patch('twinkle.utils.loader.importlib.import_module') as mock_import:
                        mock_module = MagicMock()
                        mock_module.__file__ = str(init_file)
                        # Make inspect.getmembers work correctly
                        mock_module.OtherClass = other_class
                        mock_import.return_value = mock_module
                        
                        with pytest.raises(ValueError, match="Cannot find any subclass"):
                            Plugin.load_plugin("ms://test/plugin", BasePlugin)
    
    def test_load_plugin_ms_hub(self): 
        """Test loading plugin from ModelScope hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text(
                "class BasePlugin:\n"
                "    def __init__(self, name='default'):\n"
                "        self.name = name\n\n"
                "class TestPlugin(BasePlugin):\n"
                "    pass\n"
            )
            
            # Create a mock module that matches the expected structure
            mock_module = MagicMock()
            mock_module.__file__ = str(init_file)
            test_plugin_class = type('TestPlugin', (BasePlugin,), {})
            test_plugin_class.__module__ = str(init_file)
            mock_module.TestPlugin = test_plugin_class
            
            with patch('twinkle.utils.loader.MSHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with patch('twinkle.utils.loader.importlib.import_module', return_value=mock_module):
                        plugin_cls = Plugin.load_plugin("ms://test/plugin", BasePlugin)
                        assert plugin_cls.__name__ == "TestPlugin"
                        assert issubclass(plugin_cls, BasePlugin)
    
    def test_load_plugin_hf_hub(self):
        """Test loading plugin from HuggingFace hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text(
                "class BasePlugin:\n"
                "    def __init__(self, name='default'):\n"
                "        self.name = name\n\n"
                "class TestPlugin(BasePlugin):\n"
                "    pass\n"
            )
            
            # Create a mock module that matches the expected structure
            mock_module = MagicMock()
            mock_module.__file__ = str(init_file)
            test_plugin_class = type('TestPlugin', (BasePlugin,), {})
            test_plugin_class.__module__ = str(init_file)
            mock_module.TestPlugin = test_plugin_class
            
            with patch('twinkle.utils.loader.HFHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with patch('twinkle.utils.loader.importlib.import_module', return_value=mock_module):
                        plugin_cls = Plugin.load_plugin("hf://test/plugin", BasePlugin)
                        assert plugin_cls.__name__ == "TestPlugin"
                        assert issubclass(plugin_cls, BasePlugin)
    
    def test_load_plugin_sys_path_management(self):
        """Test that plugin directory is correctly added and removed from sys.path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text(
                "class BasePlugin:\n"
                "    def __init__(self, name='default'):\n"
                "        self.name = name\n\n"
                "class TestPlugin(BasePlugin):\n"
                "    pass\n"
            )
            
            # Create a mock module
            mock_module = MagicMock()
            mock_module.__file__ = str(init_file)
            test_plugin_class = type('TestPlugin', (BasePlugin,), {})
            test_plugin_class.__module__ = str(init_file)
            mock_module.TestPlugin = test_plugin_class
            
            assert tmpdir not in sys.path
            
            with patch('twinkle.utils.loader.MSHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with patch('twinkle.utils.loader.importlib.import_module', return_value=mock_module):
                        Plugin.load_plugin("ms://test/plugin", BasePlugin)
            
            assert tmpdir not in sys.path


class TestConstructClass:
    """Test construct_class functionality."""
    
    def test_construct_class_with_instance(self):
        """Test construct_class when input is already an instance."""
        instance = SamplePlugin("test")
        result = construct_class(instance, BasePlugin, [])
        assert result is instance
    
    def test_construct_class_with_class_type(self):
        """Test construct_class when input is a class type."""
        result = construct_class(SamplePlugin, BasePlugin, [], name="test")
        assert isinstance(result, SamplePlugin)
        assert result.name == "test"
    
    def test_construct_class_with_string_name(self):
        """Test construct_class when input is a string class name."""
        # Import the current test module to access SamplePlugin
        import sys
        current_module = sys.modules[__name__]
        result = construct_class("SamplePlugin", BasePlugin, [current_module], name="test")
        assert isinstance(result, SamplePlugin)
        assert result.name == "test"
    
    def test_construct_class_with_plugin_id(self):
        """Test construct_class when input is a plugin ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text(
                "class BasePlugin:\n"
                "    def __init__(self, name='default'):\n"
                "        self.name = name\n\n"
                "class TestPlugin(BasePlugin):\n"
                "    pass\n"
            )
            
            # Create a mock module
            mock_module = MagicMock()
            mock_module.__file__ = str(init_file)
            test_plugin_class = type('TestPlugin', (BasePlugin,), {})
            test_plugin_class.__module__ = str(init_file)
            mock_module.TestPlugin = test_plugin_class
            
            with patch('twinkle.utils.loader.MSHub.download_model', return_value=tmpdir):
                with patch('twinkle.utils.loader.trust_remote_code', return_value=True):
                    with patch('twinkle.utils.loader.importlib.import_module', return_value=mock_module):
                        result = construct_class("ms://test/plugin", BasePlugin, [], name="test")
                        assert isinstance(result, BasePlugin)
                        assert result.name == "test"
    
    def test_construct_class_with_invalid_input(self):
        """Test construct_class with invalid input type."""
        result = construct_class(123, BasePlugin, [])
        assert result == 123
