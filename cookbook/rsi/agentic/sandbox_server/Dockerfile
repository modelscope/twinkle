FROM python:3.11-slim

# Every instruction has to fit on ONE line: aenv's Dockerfile parser does not
# join backslash continuations, and reports the second line as an unknown
# instruction ("Dockerfile instruction ca-certificates is not supported").

# The build runs on the AgentENV host. Ours sits behind a firewall that cannot
# reach files.pythonhosted.org (it 302s and then stalls), and ms-agent pulls in
# pandas/matplotlib/modelscope -- a stalled index is a build that never
# finishes. Point these at a different index if your host reaches pypi directly.
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

# ripgrep is not optional: file_system's `grep` uses `rg` when it is on PATH and
# silently falls back to a Python scan with a different output shape when it is
# not. The policy is trained on whatever it sees, so the sandbox has to take the
# same branch a serving deployment does.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git ripgrep && rm -rf /var/lib/apt/lists/*

# ms-agent from source, not `pip install ms-agent`: the tools the policy is
# trained against are the ones in this repository, and a released wheel can lag
# behind it.
#
# curl, not `git clone`: Debian's git links against GnuTLS, and cloning GitHub
# from inside the build VM stalls for ~15 minutes and then dies with "GnuTLS
# recv error (-110)". The same host pulls the codeload tarball over curl's
# OpenSSL at 12MB/s. Version comes from ms_agent/version.py, not scm, so the
# missing .git costs nothing.
RUN mkdir -p /opt/ms-agent && curl -fsSL https://codeload.github.com/modelscope/ms-agent/tar.gz/refs/heads/main | tar -xz -C /opt/ms-agent --strip-components=1 && pip install --no-cache-dir -e /opt/ms-agent

# One pip call, not three: aenv runs each instruction inside a fresh microVM with
# no layer cache, so every extra RUN is another full dependency resolution.
#
# httpx      -- imported by ms_agent/llm/openai_llm.py, which `import ms_agent`
#               reaches via tools/search/websearch_tool, but no requirements file
#               declares it. Without it the image builds cleanly and then every
#               sandbox fails at import.
# ipykernel,
# jupyter-client
#            -- notebook_executor pip-installs these on first call.
# the rest   -- LocalCodeExecutionTool._check_dependencies installs this exact
#               list at construction time, i.e. on every sandbox boot. ms-agent's
#               own requirements cover six of them; without the other five each
#               of the N sandboxes in a training step spends its first seconds
#               fetching seaborn, scikit-learn, beautifulsoup4, lxml and pyarrow.
#               Kept as the full list so it stays correct if ms-agent's
#               dependencies shift.
#
# In a sandbox, any of these missing is either a network round trip at the start
# of every episode or an outright failure on an air-gapped host.
RUN pip install --no-cache-dir httpx ipykernel jupyter-client numpy pandas matplotlib seaborn scikit-learn requests beautifulsoup4 lxml pillow tqdm pyarrow

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
