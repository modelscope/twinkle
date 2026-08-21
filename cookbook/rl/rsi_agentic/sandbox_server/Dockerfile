FROM python:3.11-slim

# ripgrep is not optional: file_system's `grep` uses `rg` when it is on PATH and
# silently falls back to a Python scan with a different output shape when it is
# not. The policy is trained on whatever it sees, so the sandbox has to take the
# same branch a serving deployment does.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/*

# The exact ms-agent checkout the training host imports, staged here by
# install.sh -- not `pip install ms-agent`. The host runs an editable install of
# this working copy, and a tool whose output differs by one local commit is a
# train/serve mismatch that only shows up after deployment.
COPY ms-agent /opt/ms-agent
RUN pip install --no-cache-dir -e /opt/ms-agent

# notebook_executor pip-installs these the first time it is called. In a sandbox
# that is either a network round trip at the start of every episode or an
# outright failure on an air-gapped host, so they ship in the image.
RUN pip install --no-cache-dir ipykernel jupyter-client

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
