# Client and server distributions

This directory is not a Python distribution.

```text
twinkle_cs/
├── client/       # twinkle-client -> twinkle-kit
├── server/       # twinkle-server -> twinkle-kit
└── java-client/  # reserved; no SDK code or build configuration yet
```

Build each Python distribution from the repository root:

```sh
python -m build
python -m build twinkle_cs/client
python -m build twinkle_cs/server
```

The shared `twinkle.protocol` belongs to kit. Client and server never install one
another. Their CLI entry points and package resources belong to their own wheels.
The initial Python versions are `0.4.0.dev0`; each distribution can be versioned
independently within its declared kit compatibility range.

Release tags `v<version>` (kit), `twinkle-client-v<version>` and
`twinkle-server-v<version>` publish the corresponding Python distribution.

The Tinker request encoder is now
`twinkle_client.utils.tinker_datum.input_feature_to_datum`; client examples do not
import the server. Server loggers and telemetry scopes keep their `twinkle.server.*` names so existing
log levels, handlers and telemetry queries continue to work across the source move.

Model-to-template selection is shared through `twinkle.template.mapping`; remote
Swift sampler clients can resolve template names without installing the server.
