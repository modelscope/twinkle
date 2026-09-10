# Copyright (c) ModelScope Contributors. All rights reserved.
"""Files that run *inside* a sandbox, uploaded there by ``envs.remote_tools``.

A package only so that ``pip install`` ships them; nothing here is meant to be
imported on the training host. ``server.py`` is the transport and every
``runtime_<name>.py`` beside it is one agent framework's tools -- which is the
whole of what supporting another framework takes.
"""
