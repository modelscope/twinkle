"""Generate the ACL for the dedicated Twinkle Redis instance."""

import argparse
import getpass
import hashlib
import os
import re
from pathlib import Path


def password_digest(prompt: str, env_key: str) -> str:
    password = os.environ.get(env_key) or getpass.getpass(prompt)
    if not password:
        raise SystemExit("Redis passwords must be nonempty.")
    return hashlib.sha256(password.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key-prefix", default="twinkle:h20:")
    args = parser.parse_args()
    prefix = args.key_prefix
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", prefix):
        parser.error("--key-prefix must contain only letters, digits, colons, dots, underscores or hyphens")

    runtime = password_digest("Twinkle Redis password: ", "TWINKLE_REDIS_PASSWORD")
    grafana = password_digest("Grafana Redis password: ", "OBS_REDIS_PASSWORD")
    if runtime == grafana:
        parser.error("Use different passwords for Twinkle and Grafana.")
    snapshots = " ".join(f"~{prefix}admin:index:{name}" for name in ("sessions", "trainings", "sampling"))
    with args.output.open("x") as stream:
        stream.write(
            "user default off\n"
            f"user twinkle on #{runtime} ~{prefix}* "
            "+get +mget +set +del +exists +scan +watch +unwatch +multi +exec +discard "
            "+ping +select +hello +client|setinfo\n"
            f"user twinkle-grafana on #{grafana} {snapshots} +get\n"
        )


if __name__ == "__main__":
    main()
