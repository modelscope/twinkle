# Copyright (c) ModelScope Contributors. All rights reserved.
"""HTTP middleware for the server deployments (moved here from ``server/utils/`` by R14).

Sits next to ``deployment.py``'s middleware stack: ``auth.verify_request_token`` is the
token-verification middleware registered by ``build_deployment_app``.
"""
