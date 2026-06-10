# Copyright (c) ModelScope Contributors. All rights reserved.
"""Base file manager and abstract training-run manager.

Relocated from ``utils/checkpoint_base.py`` (TIER 2 consolidation). No logic change.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from twinkle import get_logger
from .paths import SAVE_DIR_POINTER_KEY, TWINKLE_DEFAULT_SAVE_DIR, _hash_token, _resolve_client_save_dir

logger = get_logger()


class BaseFileManager:
    """Base file manager with common utilities."""

    @staticmethod
    def get_dir_size(path: Path) -> int:
        """Calculate total size of files in a directory."""
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    total += p.stat().st_size
        return total


class BaseTrainingRunManager(BaseFileManager, ABC):
    """
    Abstract base class for managing training run metadata.

    Subclasses must implement:
    - train_run_info_filename property
    - _create_training_run method
    - _training_runs_response_cls property
    """

    def __init__(self, token: str):
        """
        Initialize the manager with a user token.

        Args:
            token: User's authentication token for directory isolation
        """
        self.token = token

    @property
    @abstractmethod
    def train_run_info_filename(self) -> str:
        """Return the filename for training run metadata."""
        pass

    @abstractmethod
    def _create_training_run(self, model_id: str, run_config: Any) -> dict[str, Any]:
        """
        Create training run data from model_id and run_config.

        Args:
            model_id: The model identifier
            run_config: The run configuration

        Returns:
            Dictionary with training run data
        """
        pass

    @abstractmethod
    def _parse_training_run(self, data: dict[str, Any]) -> Any:
        """
        Parse training run data into the appropriate model.

        Args:
            data: Raw training run data

        Returns:
            TrainingRun model instance
        """
        pass

    @abstractmethod
    def _create_training_runs_response(self, runs: list[Any], limit: int, offset: int, total: int) -> Any:
        """
        Create a training runs response.

        Args:
            runs: List of training runs
            limit: Page limit
            offset: Page offset
            total: Total count

        Returns:
            TrainingRunsResponse model instance
        """
        pass

    def _token_base_dir(self, save_dir: str | None = None) -> Path:
        if save_dir:
            base_path = _resolve_client_save_dir(save_dir)
        else:
            base_path = Path(TWINKLE_DEFAULT_SAVE_DIR).absolute()
        return base_path / _hash_token(self.token)

    def _default_model_dir(self, model_id: str) -> Path:
        return self._token_base_dir() / model_id

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _read_save_dir_pointer(self, model_id: str) -> dict[str, Any] | None:
        data = self._read_json_file(self._default_model_dir(model_id) / self.train_run_info_filename)
        if data.get(SAVE_DIR_POINTER_KEY):
            return data
        return None

    @staticmethod
    def _extract_save_dir(run_config: Any) -> str | None:
        save_dir = getattr(run_config, 'save_dir', None)
        if save_dir:
            return save_dir
        model_extra = getattr(run_config, 'model_extra', None)
        if isinstance(model_extra, dict):
            save_dir = model_extra.get('save_dir')
            if save_dir:
                return save_dir
        user_metadata = getattr(run_config, 'user_metadata', None)
        if isinstance(user_metadata, dict):
            return user_metadata.get('save_dir')
        return None

    def get_base_dir(self) -> Path:
        """
        Get base directory with token-based isolation.

        The token is never written to disk in plaintext; instead a salted
        HMAC-SHA256 digest is used as the directory name so that the real
        token cannot be recovered by inspecting the filesystem.

        Returns:
            Path to token-specific base directory
        """
        return self._token_base_dir()

    def get_model_dir(self, model_id: str, save_dir: str | None = None) -> Path:
        """
        Get model directory with token-based isolation.

        Args:
            model_id: The model identifier

        Returns:
            Path to model directory
        """
        if save_dir:
            return self._token_base_dir(save_dir) / model_id

        pointer = self._read_save_dir_pointer(model_id)
        if pointer and pointer.get('save_dir'):
            return self._token_base_dir(pointer.get('save_dir')) / model_id
        return self.get_base_dir() / model_id

    def _read_info(self, model_id: str) -> dict[str, Any]:
        """
        Read training run metadata from disk.

        Args:
            model_id: The model identifier

        Returns:
            Dictionary with metadata or empty dict if not found
        """
        metadata_path = self.get_model_dir(model_id) / self.train_run_info_filename
        if not metadata_path.exists():
            return {}
        data = self._read_json_file(metadata_path)
        if data.get(SAVE_DIR_POINTER_KEY):
            save_dir = data.get('save_dir')
            if not save_dir:
                return {}
            target_path = self.get_model_dir(model_id, save_dir=save_dir) / self.train_run_info_filename
            return self._read_json_file(target_path)
        return data

    def _write_info(self, model_id: str, data: dict[str, Any]):
        """
        Write training run metadata to disk.

        Args:
            model_id: The model identifier
            data: Metadata to write
        """
        save_dir = data.get('save_dir')
        model_dir = self.get_model_dir(model_id, save_dir=save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / self.train_run_info_filename
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

        if save_dir:
            pointer_dir = self._default_model_dir(model_id)
            if pointer_dir.resolve() != model_dir.resolve():
                pointer_dir.mkdir(parents=True, exist_ok=True)
                pointer_path = pointer_dir / self.train_run_info_filename
                pointer_data = {
                    SAVE_DIR_POINTER_KEY: True,
                    'training_run_id': model_id,
                    'save_dir': save_dir,
                }
                with open(pointer_path, 'w') as f:
                    json.dump(pointer_data, f, indent=2)

    def save(self, model_id: str, run_config: Any):
        """
        Save training run metadata with token-based isolation.

        Args:
            model_id: Unique identifier for the model
            run_config: Configuration for the training run
        """
        new_data = self._create_training_run(model_id, run_config)
        save_dir = self._extract_save_dir(run_config)
        if save_dir:
            new_data['save_dir'] = _resolve_client_save_dir(save_dir).as_posix()
        self._write_info(model_id, new_data)

    def get(self, model_id: str) -> Any | None:
        """
        Get training run metadata.

        Args:
            model_id: The model identifier

        Returns:
            TrainingRun object or None if not found
        """
        data = self._read_info(model_id)
        if not data:
            return None
        return self._parse_training_run(data)

    def update(self, model_id: str, updates: dict[str, Any]):
        """
        Update training run metadata.

        Args:
            model_id: The model identifier
            updates: Dictionary of fields to update
        """
        info = self._read_info(model_id)
        if info:
            info.update(updates)
            self._write_info(model_id, info)

    def list_runs(self, limit: int = 20, offset: int = 0) -> Any:
        """
        List training runs for the current user.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            TrainingRunsResponse with list of training runs
        """
        base_dir = self.get_base_dir()
        candidates = []
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir() and (d / self.train_run_info_filename).exists():
                    candidates.append(d)

        candidates.sort(key=lambda d: (d / self.train_run_info_filename).stat().st_mtime, reverse=True)

        # All runs in the token directory belong to this user
        runs = []
        for d in candidates:
            run = self.get(d.name)
            if run:
                runs.append(run)

        total = len(runs)
        selected = runs[offset:offset + limit]

        return self._create_training_runs_response(selected, limit, offset, total)
