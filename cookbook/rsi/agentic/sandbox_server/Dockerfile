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

# apt from the same mirror, for the same reason. Measured from inside a sandbox
# on this host on 2026-08-23: deb.debian.org delivered 33 KB/s, and
# mirrors.aliyun.com/debian 5.4 MB/s -- for the ~200MB ffmpeg/imagemagick set
# that is the difference between a minute and two hours. The 9.6MB package index
# alone stalled one build long enough to look hung. Both file names are listed
# because trixie-based images carry .sources and older ones sources.list.
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true

# dpkg fsyncs every control file it unpacks, and fsync does not work in these VMs:
# probed on 2026-08-23 inside a running sandbox, os.fsync returned EIO in /, /tmp,
# /workspace and /root alike with 60GB free -- the virtual block device does not
# implement flush. Without this option each package fails to unpack with "unable
# to sync file '/var/lib/dpkg/tmp.ci//md5sums': Input/output error" and apt exits
# 100. Writing without fsync is the usual answer in a container and loses nothing
# that matters here, since the image is built once and never survives a crash.
RUN mkdir -p /etc/dpkg/dpkg.cfg.d && echo force-unsafe-io > /etc/dpkg/dpkg.cfg.d/99-unsafe-io

# ripgrep is not optional: file_system's `grep` uses `rg` when it is on PATH and
# silently falls back to a Python scan with a different output shape when it is
# not. The policy is trained on whatever it sees, so the sandbox has to take the
# same branch a serving deployment does.
#
# ffmpeg and imagemagick: episodes reach for them from shell_executor and got
# "ffmpeg: not found" (1 call) and "convert: not found" (2 calls) in ex6. Both
# are the standard answer for the media half of a task, so a sandbox without them
# turns a reasonable plan into a dead end.
#
# The rest are the everyday command-line tools a python:slim image happens not to
# carry. `zip` and `unzip` were each asked for and missing in ex7, and the wider
# list is there because the misses are a long tail -- every name recorded across
# ex3-ex7 appears once or twice, so waiting for a second sighting means paying for
# the same dead end again. Probed against the live image on 2026-08-23, all of
# these were absent. Deliberately left out as too large for what they would buy:
# libreoffice (~700MB), pandoc, build-essential, and weasyprint's pango/cairo
# stack.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl wget git ripgrep ffmpeg imagemagick zip unzip bzip2 xz-utils p7zip-full jq sqlite3 tree file bc patch dos2unix bsdextrautils xxd poppler-utils && rm -rf /var/lib/apt/lists/*

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
# openpyxl,
# reportlab,
# pdfplumber
#            -- what episodes actually asked for and did not get: 10 calls died
#               on `No module named 'openpyxl'`, 3 on reportlab, 1 on pdfplumber
#               (ex3-ex6, 983 recorded calls). Spreadsheets and PDFs are a large
#               part of what the keyword bank proposes, and pandas' Excel support
#               needs openpyxl anyway.
# pyspellchecker,
# python-docx,
# the rest of
# the tail    -- same reasoning as the apt list above: `No module named` was
#               recorded once each for weasyprint and spellchecker, and these are
#               what a task about documents, spreadsheets, archives or text
#               normally imports next. All were absent when probed against the
#               live image on 2026-08-23. weasyprint itself is not here: it needs
#               pango and cairo, which is a different size of decision.
#
# In a sandbox, any of these missing is either a network round trip at the start
# of every episode or an outright failure on an air-gapped host.
RUN pip install --no-cache-dir httpx ipykernel jupyter-client numpy pandas matplotlib seaborn scikit-learn requests beautifulsoup4 lxml pillow tqdm pyarrow openpyxl reportlab pdfplumber python-docx python-pptx xlsxwriter pypdf pymupdf toml jinja2 chardet regex tabulate sympy networkx faker pyspellchecker

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
