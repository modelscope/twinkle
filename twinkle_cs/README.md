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

## Versioning and releases

No distribution carries a version of its own: the repository-root `VERSION` file is the single
source for every distribution in this repository, Python or otherwise.

- `pyproject.toml` (twinkle-kit) reads `VERSION` directly (`[tool.setuptools.dynamic]`).
- `client/` and `server/` cannot reference the repository root from their `pyproject.toml`
  (setuptools refuses files outside the project directory), so their `setup.py` copies `VERSION`
  next to itself before calling `setup()`, and `MANIFEST.in` ships that copy inside the sdist so
  the wheel stays rebuildable from the sdist alone. The copies are git-ignored.
- At runtime `twinkle.__version__`, `twinkle_client.__version__` and `twinkle_server.__version__`
  read the installed package metadata; a source checkout falls back to the `VERSION` file.

Pushing a single `v<VERSION>` tag publishes twinkle-kit, twinkle-client and twinkle-server
together with that version (`.github/workflows/publish.yaml`, which fails if the tag and
`VERSION` disagree). A future non-Python distribution reads the same `VERSION` file and joins the
same workflow.

The Tinker request encoder is now
`twinkle_client.utils.tinker_datum.input_feature_to_datum`; client examples do not
import the server. Server loggers and telemetry scopes keep their `twinkle.server.*` names so existing
log levels, handlers and telemetry queries continue to work across the source move.

Model-to-template selection is shared through `twinkle.template.mapping`; remote
Swift sampler clients can resolve template names without installing the server.
