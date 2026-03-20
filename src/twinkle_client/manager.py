# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import atexit
import threading
from typing import Any, Dict, List, Optional, Tuple
from twinkle import get_logger
from twinkle_client.types.server import (DeleteCheckpointResponse, GetServerCapabilitiesResponse)
from twinkle_client.types.session import (CreateSessionRequest, CreateSessionResponse, SessionHeartbeatRequest,
                                           SessionHeartbeatResponse)
from twinkle_client.types.training import (Checkpoint, Cursor, ParsedCheckpointTwinklePath, TrainingRun,
                                            TrainingRunsResponse, WeightsInfoResponse)
from .http import get_api_key, get_base_url, http_delete, http_get, http_post, set_api_key, set_base_url, set_session_id

logger = get_logger()

class TwinkleClientError(Exception):
    """Base exception for TwinkleManager errors."""
    pass


class TwinkleClient:
    """
    Client manager for interacting with Twinkle REST API.

    On initialization this client:
    - Sets the base_url and api_key into the shared context so that all other
      client objects (MultiLoraTransformersModel, vLLMSampler, processor clients)
      automatically pick up the same configuration.
    - Creates a server-side session and stores the session_id in context so that
      every outgoing HTTP request carries it in the ``X-Twinkle-Session-Id`` header.
    - Starts a lightweight background thread that touches the session every
      ``session_heartbeat_interval`` seconds to keep it alive.

    Args:
        base_url: Base URL of the Twinkle server (e.g. "http://localhost:8000").
                  Falls back to the ``TWINKLE_SERVER_URL`` environment variable.
        api_key: API key for authentication.  Falls back to the
                 ``TWINKLE_SERVER_TOKEN`` environment variable.
        route_prefix: API route prefix (default: "/twinkle").
        session_heartbeat_interval: Seconds between session touch calls (default: 30).
        session_metadata: Optional metadata dict stored with the session on the server.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        route_prefix: Optional[str] = '/twinkle',
        session_heartbeat_interval: int = 10,
        session_metadata: Optional[Dict[str, Any]] = None,
    ):
        # Resolve and store config, then propagate to context so all generated
        # client objects that call get_base_url() / get_api_key() get these values.
        if base_url:
            set_base_url(base_url)
        if api_key:
            set_api_key(api_key)

        self.base_url = get_base_url()
        self.api_key = get_api_key()
        self.route_prefix = route_prefix.rstrip('/') if route_prefix else ''

        # Create a server-side session.
        self._session_id: str = self.create_session(session_metadata)
        set_session_id(self._session_id)

        # Start background session-touch thread.
        self._heartbeat_interval = session_heartbeat_interval
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._touch_session_loop,
            daemon=True,
            name='TwinkleSessionHeartbeat',
        )
        self._heartbeat_thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a server-side session.

        Args:
            metadata: Optional metadata dict stored with the session on the server.

        Returns:
            The session ID string.

        Raises:
            TwinkleClientError: If the session creation request fails.
        """
        resp = http_post(
            self._get_url('/create_session'),
            json_data=CreateSessionRequest(metadata=metadata).model_dump(),
        )
        resp.raise_for_status()
        return CreateSessionResponse(**resp.json()).session_id

    def _touch_session_loop(self) -> None:
        """Background loop: touch the session every ``_heartbeat_interval`` seconds.

        Uses a fixed-rate design: the wall-clock period between successive
        server-side heartbeats stays close to ``_heartbeat_interval`` regardless
        of how long the HTTP call takes, by subtracting elapsed time from the
        subsequent sleep.
        """
        import time
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            success = False
            try:
                logger.debug(f'[TwinkleClient] Touching session (session={self._session_id})...')
                resp = http_post(
                    self._get_url('/session_heartbeat'),
                    json_data=SessionHeartbeatRequest(session_id=self._session_id).model_dump(),
                    timeout=min(self._heartbeat_interval, 10),
                )
                resp.raise_for_status()
                success = True
            except Exception as e:
                logger.error(f'[TwinkleClient] Session heartbeat error: {e}')
            elapsed = time.monotonic() - t0
            if success:
                logger.debug(f'[TwinkleClient] Session heartbeat OK (elapsed={elapsed:.2f}s)')
            sleep_time = max(0.0, self._heartbeat_interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def close(self) -> None:
        """Stop the background heartbeat thread and clear session context."""
        self._stop_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Check if the Twinkle server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = http_get(self._get_url('/healthz'))
            return response.status_code == 200
        except Exception:
            return False

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        """
        Get the server's supported models and capabilities.

        Returns:
            :class:`~twinkle_client.types.server.GetServerCapabilitiesResponse` with
            ``supported_models`` field containing a list of supported model names.

        Raises:
            TwinkleClientError: If the request fails.
        """
        response = http_get(self._get_url('/get_server_capabilities'))
        data = self._handle_response(response)
        return GetServerCapabilitiesResponse(**data)

    # ------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------

    def list_training_runs(self, limit: int = 20, offset: int = 0, all_users: bool = False) -> List[TrainingRun]:
        """
        List training runs.

        By default, only returns training runs owned by the current user.

        Args:
            limit: Maximum number of results (default: 20).
            offset: Offset for pagination (default: 0).
            all_users: If True, return all runs (if permission allows).

        Returns:
            List of :class:`~twinkle_client.types.training.TrainingRun` objects.

        Raises:
            TwinkleClientError: If the request fails.
        """
        params: Dict[str, Any] = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = http_get(self._get_url('/training_runs'), params=params)
        data = self._handle_response(response)

        return [TrainingRun(**r) for r in data.get('training_runs', [])]

    def list_training_runs_with_cursor(
        self,
        limit: int = 20,
        offset: int = 0,
        all_users: bool = False,
    ) -> Tuple[List[TrainingRun], Cursor]:
        """
        List training runs with pagination info.

        Args:
            limit: Maximum number of results (default: 20).
            offset: Offset for pagination (default: 0).
            all_users: If True, return all runs (if permission allows).

        Returns:
            Tuple of (list of TrainingRun, Cursor with pagination info).

        Raises:
            TwinkleClientError: If the request fails.
        """
        params: Dict[str, Any] = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = http_get(self._get_url('/training_runs'), params=params)
        data = self._handle_response(response)

        runs = [TrainingRun(**r) for r in data.get('training_runs', [])]
        cursor = Cursor(**data.get('cursor', {}))
        return runs, cursor

    def get_training_run(self, run_id: str) -> TrainingRun:
        """
        Get details of a specific training run.

        Args:
            run_id: The training run identifier.

        Returns:
            :class:`~twinkle_client.types.training.TrainingRun` object with run details.

        Raises:
            TwinkleClientError: If run not found or access denied.
        """
        response = http_get(self._get_url(f'/training_runs/{run_id}'))
        data = self._handle_response(response)
        return TrainingRun(**data)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """
        List checkpoints for a training run.

        Args:
            run_id: The training run identifier.

        Returns:
            List of :class:`~twinkle_client.types.training.Checkpoint` objects.

        Raises:
            TwinkleClientError: If run not found or access denied.
        """
        response = http_get(self._get_url(f'/training_runs/{run_id}/checkpoints'))
        data = self._handle_response(response)
        return [Checkpoint(**c) for c in data.get('checkpoints', [])]

    def get_checkpoint_path(self, run_id: str, checkpoint_id: str) -> ParsedCheckpointTwinklePath:
        """
        Get the filesystem path and twinkle:// path for a checkpoint.

        Args:
            run_id: The training run identifier.
            checkpoint_id: The checkpoint identifier (e.g. "weights/20240101_120000").

        Returns:
            :class:`~twinkle_client.types.training.ParsedCheckpointTwinklePath` with
            ``path`` (filesystem) and ``twinkle_path`` fields.

        Raises:
            TwinkleClientError: If checkpoint not found or access denied.
        """
        response = http_get(self._get_url(f'/checkpoint_path/{run_id}/{checkpoint_id}'))
        data = self._handle_response(response)
        return ParsedCheckpointTwinklePath(
            path=data.get('path', ''),
            twinkle_path=data.get('twinkle_path', ''),
            training_run_id=run_id,
            checkpoint_type=checkpoint_id.split('/')[0] if '/' in checkpoint_id else '',
            checkpoint_id=checkpoint_id,
        )

    def get_checkpoint_twinkle_path(self, run_id: str, checkpoint_id: str) -> str:
        """
        Get the twinkle:// path for a checkpoint.

        Args:
            run_id: The training run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            Twinkle path string (e.g. "twinkle://run_id/weights/checkpoint_name").

        Raises:
            TwinkleClientError: If checkpoint not found or access denied.
        """
        return self.get_checkpoint_path(run_id, checkpoint_id).twinkle_path

    def delete_checkpoint(self, run_id: str, checkpoint_id: str) -> DeleteCheckpointResponse:
        """
        Delete a checkpoint.

        Args:
            run_id: The training run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            :class:`~twinkle_client.types.server.DeleteCheckpointResponse` indicating success.

        Raises:
            TwinkleClientError: If checkpoint not found or access denied.
        """
        url = self._get_url(f'/training_runs/{run_id}/checkpoints/{checkpoint_id}')
        response = http_delete(url)
        data = self._handle_response(response)
        return DeleteCheckpointResponse(**data)

    # ------------------------------------------------------------------
    # Weights Info
    # ------------------------------------------------------------------

    def get_weights_info(self, twinkle_path: str) -> WeightsInfoResponse:
        """
        Get information about saved weights.

        Args:
            twinkle_path: The twinkle:// path to the weights.

        Returns:
            :class:`~twinkle_client.types.training.WeightsInfoResponse` with fields:
            ``training_run_id``, ``base_model``, ``model_owner``, ``is_lora``, ``lora_rank``.

        Raises:
            TwinkleClientError: If weights not found or access denied.
        """
        response = http_post(self._get_url('/weights_info'), json_data={'twinkle_path': twinkle_path})
        data = self._handle_response(response)
        return WeightsInfoResponse(**data)

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def get_latest_checkpoint_path(self, run_id: str) -> Optional[str]:
        """
        Get the filesystem path to the latest checkpoint for a training run.

        Useful for resume training — returns the path to the most recent checkpoint.

        Args:
            run_id: The training run identifier.

        Returns:
            Filesystem path string to the latest checkpoint, or ``None`` if none exist.

        Raises:
            TwinkleClientError: If run not found or access denied.
        """
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        return self.get_checkpoint_path(run_id, latest.checkpoint_id).path

    def find_training_run_by_model(self, base_model: str) -> List[TrainingRun]:
        """
        Find training runs for a specific base model.

        Args:
            base_model: The base model name to search for.

        Returns:
            List of :class:`~twinkle_client.types.training.TrainingRun` objects
            matching the base model.
        """
        all_runs = self.list_training_runs(limit=100)
        return [run for run in all_runs if run.base_model == base_model]
