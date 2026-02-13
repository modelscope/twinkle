# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Reuse Pydantic models from server
from twinkle.server.twinkle.common.io_utils import Checkpoint, Cursor, TrainingRun
from .http.http_utils import http_get, http_post


class TwinkleClientError(Exception):
    """Base exception for TwinkleManager errors."""
    pass


class TwinkleClient:
    """
    Client manager for interacting with Twinkle REST API.

    This manager provides methods to:
    - List training runs owned by the current user
    - Get details of specific training runs
    - List checkpoints for a training run
    - Get checkpoint file paths for resume training
    - Delete checkpoints

    All operations respect user permissions - users can only access
    and modify their own resources.

    Args:
        base_url: Base URL of the Twinkle server (e.g., "http://localhost:8000").
        api_key: API key for authentication. If not provided, uses
                 TWINKLE_SERVER_TOKEN environment variable
        route_prefix: API route prefix (default: "/server")
    """

    def __init__(self, base_url: str = None, api_key: str = None, route_prefix: str | None = '/server'):
        self.base_url = base_url
        self.api_key = api_key
        self.route_prefix = route_prefix.rstrip('/') if route_prefix else ''

    def _get_url(self, endpoint: str) -> str:
        """Construct full URL for an endpoint."""
        return f'{self.base_url}{self.route_prefix}{endpoint}'

    def _handle_response(self, response, expected_code: int = 200) -> dict[str, Any]:
        """Handle HTTP response and raise appropriate errors."""
        if response.status_code != expected_code:
            try:
                error_data = response.json()
                detail = error_data.get('detail', str(error_data))
            except Exception:
                detail = response.text
            raise TwinkleClientError(f'Request failed with status {response.status_code}: {detail}')
        return response.json()

    # ----- Health Check -----

    def health_check(self) -> bool:
        """
        Check if the Twinkle server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = http_get(self._get_url('/healthz'))
            return response.status_code == 200
        except Exception:
            return False

    # ----- Training Runs -----

    def list_training_runs(self, limit: int = 20, offset: int = 0, all_users: bool = False) -> list[TrainingRun]:
        """
        List training runs.

        By default, only returns training runs owned by the current user.

        Args:
            limit: Maximum number of results (default: 20)
            offset: Offset for pagination (default: 0)
            all_users: If True, return all runs (if permission allows)

        Returns:
            List of TrainingRun objects

        Raises:
            TwinkleManagerError: If the request fails
        """
        params = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = http_get(self._get_url('/training_runs'), params=params)
        data = self._handle_response(response)

        runs = []
        for run_data in data.get('training_runs', []):
            runs.append(TrainingRun(**run_data))
        return runs

    def list_training_runs_with_cursor(self,
                                       limit: int = 20,
                                       offset: int = 0,
                                       all_users: bool = False) -> tuple[list[TrainingRun], Cursor]:
        """
        List training runs with pagination info.

        Args:
            limit: Maximum number of results (default: 20)
            offset: Offset for pagination (default: 0)
            all_users: If True, return all runs (if permission allows)

        Returns:
            Tuple of (list of TrainingRun, Cursor with pagination info)

        Raises:
            TwinkleManagerError: If the request fails
        """
        params = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = http_get(self._get_url('/training_runs'), params=params)
        data = self._handle_response(response)

        runs = []
        for run_data in data.get('training_runs', []):
            runs.append(TrainingRun(**run_data))

        cursor = Cursor(**data.get('cursor', {}))
        return runs, cursor

    def get_training_run(self, run_id: str) -> TrainingRun:
        """
        Get details of a specific training run.

        Args:
            run_id: The training run identifier

        Returns:
            TrainingRun object with run details

        Raises:
            TwinkleManagerError: If run not found or access denied
        """
        response = http_get(self._get_url(f'/training_runs/{run_id}'))
        data = self._handle_response(response)
        return TrainingRun(**data)

    # ----- Checkpoints -----

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """
        List checkpoints for a training run.

        Args:
            run_id: The training run identifier

        Returns:
            List of Checkpoint objects

        Raises:
            TwinkleManagerError: If run not found or access denied
        """
        response = http_get(self._get_url(f'/training_runs/{run_id}/checkpoints'))
        data = self._handle_response(response)

        checkpoints = []
        for ckpt_data in data.get('checkpoints', []):
            checkpoints.append(Checkpoint(**ckpt_data))
        return checkpoints

    def get_checkpoint_path(self, run_id: str, checkpoint_id: str) -> str:
        """
        Get the filesystem path for a checkpoint.

        This path can be used to load weights for resume training.

        Args:
            run_id: The training run identifier
            checkpoint_id: The checkpoint identifier (e.g., "weights/20240101_120000")

        Returns:
            Filesystem path to the checkpoint directory

        Raises:
            TwinkleManagerError: If checkpoint not found or access denied
        """
        response = http_get(self._get_url(f'/checkpoint_path/{run_id}/{checkpoint_id}'))
        data = self._handle_response(response)
        return data.get('path', '')

    def get_checkpoint_twinkle_path(self, run_id: str, checkpoint_id: str) -> str:
        """
        Get the twinkle:// path for a checkpoint.

        Args:
            run_id: The training run identifier
            checkpoint_id: The checkpoint identifier

        Returns:
            Twinkle path (e.g., "twinkle://run_id/weights/checkpoint_name")

        Raises:
            TwinkleManagerError: If checkpoint not found or access denied
        """
        response = http_get(self._get_url(f'/checkpoint_path/{run_id}/{checkpoint_id}'))
        data = self._handle_response(response)
        return data.get('twinkle_path', '')

    def delete_checkpoint(self, run_id: str, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            run_id: The training run identifier
            checkpoint_id: The checkpoint identifier

        Returns:
            True if deletion was successful

        Raises:
            TwinkleManagerError: If checkpoint not found or access denied
        """
        from .http import http_delete

        url = self._get_url(f'/training_runs/{run_id}/checkpoints/{checkpoint_id}')
        response = http_delete(url)
        data = self._handle_response(response)
        return data.get('success', False)

    # ----- Weights Info -----

    def get_weights_info(self, twinkle_path: str) -> dict[str, Any]:
        """
        Get information about saved weights.

        Args:
            twinkle_path: The twinkle:// path to the weights

        Returns:
            Dictionary with weight information including:
            - training_run_id
            - base_model
            - model_owner
            - is_lora
            - lora_rank

        Raises:
            TwinkleManagerError: If weights not found or access denied
        """
        response = http_post(self._get_url('/weights_info'), json_data={'twinkle_path': twinkle_path})
        return self._handle_response(response)

    # ----- Convenience Methods for Resume Training -----

    def get_latest_checkpoint_path(self, run_id: str) -> str | None:
        """
        Get the path to the latest checkpoint for a training run.

        This is useful for resume training - it returns the path to the
        most recent checkpoint that can be loaded.

        Args:
            run_id: The training run identifier

        Returns:
            Filesystem path to the latest checkpoint, or None if no checkpoints exist

        Raises:
            TwinkleManagerError: If run not found or access denied
        """
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None

        # Checkpoints are sorted by time, so last one is the latest
        latest = checkpoints[-1]
        return self.get_checkpoint_path(run_id, latest.checkpoint_id)

    def find_training_run_by_model(self, base_model: str) -> list[TrainingRun]:
        """
        Find training runs for a specific base model.

        Args:
            base_model: The base model name to search for

        Returns:
            List of TrainingRun objects matching the base model
        """
        all_runs = self.list_training_runs(limit=100)
        return [run for run in all_runs if run.base_model == base_model]
