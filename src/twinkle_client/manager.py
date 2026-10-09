# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import atexit
import os
import threading
from dataclasses import replace
from typing import Any

from twinkle import get_logger
from twinkle.protocol.types.server import CapacityInfoResponse, DeleteCheckpointResponse, GetServerCapabilitiesResponse
from twinkle.protocol.types.session import CreateSessionRequest, CreateSessionResponse, SessionHeartbeatRequest
from twinkle.protocol.types.training import (Checkpoint, Cursor, ParsedCheckpointTwinklePath, TrainingRun,
                                             WeightsInfoResponse)
from twinkle_client.exceptions import TwinkleHTTPError
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.http.context import (TWINKLE_SERVER_TOKEN, TWINKLE_SERVER_URL, clear_default_transport,
                                         set_default_transport)

logger = get_logger()


class TwinkleClient:
    """Owner of one connected transport, remote session, and heartbeat thread.

    Use :meth:`connect` (normally through ``init_twinkle_client``) for remote I/O.
    ``__init__`` only accepts already-established state, which keeps partial
    connection failures from publishing a client or leaking a heartbeat thread.
    """

    def __init__(
        self,
        *,
        transport: ClientTransport,
        heartbeat_transport: ClientTransport | None = None,
        route_prefix: str = '/twinkle',
        session_heartbeat_interval: int = 10,
    ) -> None:
        """Build an already-connected client without performing remote I/O."""
        self._transport = transport
        self._heartbeat_transport = heartbeat_transport or transport
        self.base_url = transport.context.base_url
        self.api_key = transport.context.api_key
        self.route_prefix = route_prefix.rstrip('/') if route_prefix else ''
        self._session_id = transport.context.session_id
        self._heartbeat_interval = session_heartbeat_interval
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._close_lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(
        cls,
        base_url: str | None = None,
        api_key: str | None = None,
        route_prefix: str | None = '/twinkle',
        session_heartbeat_interval: int = 10,
        session_metadata: dict[str, Any] | None = None,
    ) -> TwinkleClient:
        """Create the remote session and atomically publish a connected client."""
        context = ClientContext(
            base_url=base_url or os.environ.get('TWINKLE_SERVER_URL', TWINKLE_SERVER_URL),
            api_key=api_key or os.environ.get('TWINKLE_SERVER_TOKEN', TWINKLE_SERVER_TOKEN),
        )
        transport = ClientTransport(context)
        prefix = route_prefix.rstrip('/') if route_prefix else ''
        client = None
        heartbeat_transport = None
        try:
            response = transport.post(
                f'{context.base_url}{prefix}/create_session',
                json_data=CreateSessionRequest(metadata=session_metadata).model_dump(),
            )
            session_id = CreateSessionResponse.model_validate(response.json()).session_id
            transport.bind_context(replace(context, session_id=session_id))
            heartbeat_transport = ClientTransport(transport.context)
            client = cls(
                transport=transport,
                heartbeat_transport=heartbeat_transport,
                route_prefix=prefix,
                session_heartbeat_interval=session_heartbeat_interval,
            )
            set_default_transport(transport)
            client._start_heartbeat()
            atexit.register(client.close)
            return client
        except BaseException:
            if client is None:
                if heartbeat_transport is not None:
                    heartbeat_transport.close()
                transport.close()
            else:
                client.close()
            raise

    @property
    def transport(self) -> ClientTransport:
        return self._transport

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._touch_session_loop,
            daemon=True,
            name='TwinkleSessionHeartbeat',
        )
        self._heartbeat_thread.start()

    def get_capacity_info(self) -> CapacityInfoResponse:
        """
        Get the server's global LoRA capacity information.

        Returns:
            :class:`~twinkle.protocol.types.server.CapacityInfoResponse` with
            ``max_loras``, ``used_loras``, and ``free_loras`` fields.

        Raises:
            TwinkleHTTPError: If the request fails.
        """
        response = self._transport.get(self._get_url('/capacity_info'))
        data = response.json()
        return CapacityInfoResponse(**data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_url(self, endpoint: str) -> str:
        """Construct full URL for an endpoint."""
        return f'{self.base_url}{self.route_prefix}{endpoint}'

    def create_session(self, metadata: dict[str, Any] | None = None) -> str:
        """
        Create a server-side session.

        Args:
            metadata: Optional metadata dict stored with the session on the server.

        Returns:
            The session ID string.

        Raises:
            TwinkleHTTPError: If the session creation request fails.
        """
        resp = self._transport.post(
            self._get_url('/create_session'),
            json_data=CreateSessionRequest(metadata=metadata).model_dump(),
        )
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
                self._heartbeat_transport.post(
                    self._get_url('/session_heartbeat'),
                    json_data=SessionHeartbeatRequest(session_id=self._session_id).model_dump(),
                    timeout=min(self._heartbeat_interval, 10),
                )
                success = True
            except Exception as e:
                logger.error(f'[TwinkleClient] Session heartbeat error: {e}')
            elapsed = time.monotonic() - t0
            if success:
                logger.debug(f'[TwinkleClient] Session heartbeat OK (elapsed={elapsed:.2f}s)')
            sleep_time = max(0.0, self._heartbeat_interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def close(self) -> None:
        """Stop owned resources exactly once without affecting another client."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=max(2, min(self._heartbeat_interval, 10)))
        clear_default_transport(self._transport)
        if self._heartbeat_transport is not self._transport:
            self._heartbeat_transport.close()
        self._transport.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self) -> TwinkleClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def model(self, model_id: str, **kwargs: Any):
        """Create a remote training model bound explicitly to this client."""
        from twinkle_client.model import MultiLoraTransformersModel
        return MultiLoraTransformersModel(model_id, transport=self._transport, **kwargs)

    def sampler(self, model_id: str, **kwargs: Any):
        """Create a remote sampler bound explicitly to this client."""
        from twinkle_client.sampler import vLLMSampler
        return vLLMSampler(model_id, transport=self._transport, **kwargs)

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
            response = self._transport.get(self._get_url('/healthz'))
            return response.status_code == 200
        except Exception:
            return False

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        """
        Get the server's supported models and capabilities.

        Returns:
            :class:`~twinkle.protocol.types.server.GetServerCapabilitiesResponse` with
            ``supported_models`` field containing a list of supported model names.

        Raises:
            TwinkleHTTPError: If the request fails.
        """
        cached = self._transport.cached_capabilities
        if isinstance(cached, GetServerCapabilitiesResponse):
            return cached
        response = self._transport.get(self._get_url('/get_server_capabilities'))
        capabilities = GetServerCapabilitiesResponse.model_validate(response.json())
        self._transport.cached_capabilities = capabilities
        return capabilities

    # ------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------

    def list_training_runs(self, limit: int = 20, offset: int = 0, all_users: bool = False) -> list[TrainingRun]:
        """
        List training runs.

        By default, only returns training runs owned by the current user.

        Args:
            limit: Maximum number of results (default: 20).
            offset: Offset for pagination (default: 0).
            all_users: If True, return all runs (if permission allows).

        Returns:
            List of :class:`~twinkle.protocol.types.training.TrainingRun` objects.

        Raises:
            TwinkleHTTPError: If the request fails.
        """
        params: dict[str, Any] = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = self._transport.get(self._get_url('/training_runs'), params=params)
        data = response.json()

        return [TrainingRun(**r) for r in data.get('training_runs', [])]

    def list_training_runs_with_cursor(
        self,
        limit: int = 20,
        offset: int = 0,
        all_users: bool = False,
    ) -> tuple[list[TrainingRun], Cursor]:
        """
        List training runs with pagination info.

        Args:
            limit: Maximum number of results (default: 20).
            offset: Offset for pagination (default: 0).
            all_users: If True, return all runs (if permission allows).

        Returns:
            Tuple of (list of TrainingRun, Cursor with pagination info).

        Raises:
            TwinkleHTTPError: If the request fails.
        """
        params: dict[str, Any] = {'limit': limit, 'offset': offset}
        if all_users:
            params['all_users'] = 'true'

        response = self._transport.get(self._get_url('/training_runs'), params=params)
        data = response.json()

        runs = [TrainingRun(**r) for r in data.get('training_runs', [])]
        cursor = Cursor(**data.get('cursor', {}))
        return runs, cursor

    def get_training_run(self, run_id: str) -> TrainingRun:
        """
        Get details of a specific training run.

        Args:
            run_id: The training run identifier.

        Returns:
            :class:`~twinkle.protocol.types.training.TrainingRun` object with run details.

        Raises:
            TwinkleHTTPError: If run not found or access denied.
        """
        response = self._transport.get(self._get_url(f'/training_runs/{run_id}'))
        data = response.json()
        return TrainingRun(**data)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """
        List checkpoints for a training run.

        Args:
            run_id: The training run identifier.

        Returns:
            List of :class:`~twinkle.protocol.types.training.Checkpoint` objects.

        Raises:
            TwinkleHTTPError: If run not found or access denied.
        """
        response = self._transport.get(self._get_url(f'/training_runs/{run_id}/checkpoints'))
        data = response.json()
        return [Checkpoint(**c) for c in data.get('checkpoints', [])]

    def get_checkpoint_path(self, run_id: str, checkpoint_id: str) -> ParsedCheckpointTwinklePath:
        """
        Get the filesystem path and twinkle:// path for a checkpoint.

        Args:
            run_id: The training run identifier.
            checkpoint_id: The checkpoint identifier (e.g. "weights/20240101_120000").

        Returns:
            :class:`~twinkle.protocol.types.training.ParsedCheckpointTwinklePath` with
            ``path`` (filesystem) and ``twinkle_path`` fields.

        Raises:
            TwinkleHTTPError: If checkpoint not found or access denied.
        """
        response = self._transport.get(self._get_url(f'/checkpoint_path/{run_id}/{checkpoint_id}'))
        data = response.json()
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
            TwinkleHTTPError: If checkpoint not found or access denied.
        """
        return self.get_checkpoint_path(run_id, checkpoint_id).twinkle_path

    def delete_checkpoint(self, run_id: str, checkpoint_id: str) -> DeleteCheckpointResponse:
        """
        Delete a checkpoint.

        Args:
            run_id: The training run identifier.
            checkpoint_id: The checkpoint identifier.

        Returns:
            :class:`~twinkle.protocol.types.server.DeleteCheckpointResponse` indicating success.

        Raises:
            TwinkleHTTPError: If checkpoint not found or access denied.
        """
        url = self._get_url(f'/training_runs/{run_id}/checkpoints/{checkpoint_id}')
        response = self._transport.delete(url)
        data = response.json()
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
            :class:`~twinkle.protocol.types.training.WeightsInfoResponse` with fields:
            ``training_run_id``, ``base_model``, ``model_owner``, ``is_lora``, ``lora_rank``.

        Raises:
            TwinkleHTTPError: If weights not found or access denied.
        """
        response = self._transport.post(self._get_url('/weights_info'), json_data={'twinkle_path': twinkle_path})
        data = response.json()
        return WeightsInfoResponse(**data)

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def get_latest_checkpoint_path(self, run_id: str) -> str | None:
        """
        Get the filesystem path to the latest checkpoint for a training run.

        Useful for resume training — returns the path to the most recent checkpoint.

        Args:
            run_id: The training run identifier.

        Returns:
            Filesystem path string to the latest checkpoint, or ``None`` if none exist.

        Raises:
            TwinkleHTTPError: If run not found or access denied.
        """
        checkpoints = self.list_checkpoints(run_id)
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        return self.get_checkpoint_path(run_id, latest.checkpoint_id).path

    def find_training_run_by_model(self, base_model: str) -> list[TrainingRun]:
        """
        Find training runs for a specific base model.

        Args:
            base_model: The base model name to search for.

        Returns:
            List of :class:`~twinkle.protocol.types.training.TrainingRun` objects
            matching the base model.
        """
        all_runs = self.list_training_runs(limit=100)
        return [run for run in all_runs if run.base_model == base_model]
