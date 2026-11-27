# Copyright (c) ModelScope Contributors. All rights reserved.
"""Submit a checkpoint upload and reattach retries to its recorded future."""

import asyncio
import hashlib
import hmac
import logging
import uuid

import json

from twinkle.hub import HubOperation
from twinkle.protocol.types.lifecycle import TaskEnvelope
from twinkle.protocol.types.model import UploadToHubRequest, UploadToHubResponse
from twinkle_server.checkpoint import create_checkpoint_manager
from twinkle_server.exceptions import RequestRejectedError, ResourceNotFoundError
from twinkle_server.lifecycle.envelope import envelope_from_record

logger = logging.getLogger(__name__.replace('twinkle_server', 'twinkle.server', 1))


async def submit_upload(router, token: str, body: UploadToHubRequest) -> TaskEnvelope:
    fingerprint = None
    if body.client_upload_id is None:
        request_id = f"req_{uuid.uuid4().hex}"
    else:
        # Scope the stable ID to the caller across sessions and replicas.
        identity = hmac.digest(token.encode(), body.client_upload_id.encode(), "sha256")
        request_id = f"req_upload_{identity.hex()}"
        payload = json.dumps([body.checkpoint_dir, body.hub_model_id])
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()

    async def upload():
        manager = create_checkpoint_manager(token, client_type="twinkle")
        folder = manager.resolve_upload_dir(body.checkpoint_dir)
        loop = asyncio.get_running_loop()

        async def record_progress(progress):
            try:
                async with asyncio.timeout(5):
                    await router.state.report_future_progress(request_id, progress)
            except Exception:
                # Progress is optional. Only the Hub result decides the upload outcome.
                logger.warning("Could not record progress for %s", request_id, exc_info=True)

        await record_progress({"stage": "preparing"})

        def report(**progress):
            update = record_progress(progress)
            try:
                write = asyncio.run_coroutine_threadsafe(update, loop)
            except RuntimeError:
                update.close()
                logger.warning("Could not schedule progress for %s", request_id, exc_info=True)
                return
            try:
                write.result(timeout=5)
            except Exception:
                write.cancel()
                logger.warning("Could not finish recording progress for %s", request_id, exc_info=True)

        revision = await router.call_backend(
            HubOperation.upload_checkpoint,
            repo_id=body.hub_model_id,
            folder_path=folder,
            token=body.hub_token or token,
            report=report,
        )
        return UploadToHubResponse(repo_id=HubOperation.remove_source_type(body.hub_model_id), revision=revision)

    record = await router.state.get_future(request_id) if fingerprint is not None else None
    if record is None:
        await router.schedule_background_task(
            upload, task_type="upload_to_hub", token=token, request_id=request_id, submission_fingerprint=fingerprint)
        record = await router.state.get_future(request_id)
    if fingerprint is not None and record is not None:
        if record.get("owner_token") != token:
            raise ResourceNotFoundError("Unknown future")
        if record.get("submission_fingerprint") != fingerprint:
            raise RequestRejectedError("client_upload_id is already bound to a different checkpoint or repository")
    return envelope_from_record(request_id, record)
