# twinkle-server

HTTP server for Twinkle training, sampling and data processing.

Install from this checkout (run at the repository root):

```sh
python -m pip install -e . -e ./twinkle_cs/server
```

This distribution depends on `twinkle-kit`, including its training dependencies.
It does not depend on `twinkle-client`.
The shared HTTP contract is provided by `twinkle.protocol` in `twinkle-kit`.

Python import: `twinkle_server`. Command: `twinkle-server`.
