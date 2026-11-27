# twinkle-client

Python SDK for calling Twinkle HTTP services.

Install from this checkout (run at the repository root):

```sh
python -m pip install -e . -e ./twinkle_cs/client
```

This distribution depends on `twinkle-kit`, including its training dependencies.
It does not depend on `twinkle-server`.
The shared HTTP contract is provided by `twinkle.protocol` in `twinkle-kit`.

Python import: `twinkle_client`. Command: `twinkle-auto`.
