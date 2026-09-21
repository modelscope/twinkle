# Copyright (c) ModelScope Contributors. All rights reserved.
"""HTTP middleware for the server deployments.

Sits next to ``deployment.py``'s middleware stack: ``auth.verify_request_token`` is the
token-verification middleware registered by ``build_deployment_app``.
"""
