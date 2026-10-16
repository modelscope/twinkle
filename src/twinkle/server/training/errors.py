# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError


class CapacityUnavailableError(RequestRejectedError):
    error_code = 409


class TrainingExpiredError(ResourceNotFoundError):
    pass


class ProtocolMismatchError(RequestRejectedError):
    pass
