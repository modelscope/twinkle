#!/bin/sh
set -eu

NAME="${NAME:-twinkle-rsi-msagent}"
# BASE_IMAGE="python:3.11-slim"
BASE_IMAGE="${BASE_IMAGE:-docker.m.daocloud.io/library/python:3.11-slim}"
# 65536 is not a preference: `aenv start --cold` refuses a virtual size smaller
# than the base image's ("shrinking is disabled"), and that base is 64GiB.
DISK_MB="${DISK_MB:-65536}"
CPU="${CPU:-2}"
MEMORY_MB="${MEMORY_MB:-2048}"
TTL="${TTL:-3600}"

echo "==> Starting a sandbox from $BASE_IMAGE"
SID=$(aenv start --cold "$BASE_IMAGE" -d --timeout "$TTL" \
    --cpu "$CPU" --memory "$MEMORY_MB" --disk-size-mb "$DISK_MB" | tail -1 | tr -d '\r')
echo "    sandbox $SID"

SETUP=$(cat <<'SCRIPT'
set -eux
export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com

sed -i 's|deb.debian.org|mirrors.aliyun.com|g' \
    /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list 2>/dev/null || true

# dpkg fsyncs each control file it unpacks, and fsync does not work in this VM:
# probed on 2026-08-23, os.fsync returned EIO in /, /tmp, /workspace and /root
# alike, with 60GB free -- the virtual block device simply does not implement
# flush. Without this every package fails to unpack ("unable to sync file
# '/var/lib/dpkg/tmp.ci//md5sums': Input/output error", 278 of them). The option
# tells dpkg to write without fsyncing, which is the usual answer in a container
# and costs nothing here: the sandbox is disposable and the snapshot is taken
# from the filesystem afterwards, not from the block device's write cache.
mkdir -p /etc/dpkg/dpkg.cfg.d
echo force-unsafe-io > /etc/dpkg/dpkg.cfg.d/99-unsafe-io

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl wget git ripgrep \
    ffmpeg imagemagick zip unzip bzip2 xz-utils p7zip-full jq sqlite3 tree file \
    bc patch dos2unix bsdextrautils xxd poppler-utils
rm -rf /var/lib/apt/lists/*

mkdir -p /opt/ms-agent
curl -fsSL https://codeload.github.com/modelscope/ms-agent/tar.gz/refs/heads/main \
    | tar -xz -C /opt/ms-agent --strip-components=1
pip install --no-cache-dir -e /opt/ms-agent

pip install --no-cache-dir httpx ipykernel jupyter-client numpy pandas matplotlib \
    seaborn scikit-learn requests beautifulsoup4 lxml pillow tqdm pyarrow \
    openpyxl reportlab pdfplumber python-docx python-pptx xlsxwriter pypdf \
    pymupdf toml jinja2 chardet regex tabulate sympy networkx faker pyspellchecker

mkdir -p /workspace
printf '[global]\nindex-url = %s\ntrusted-host = %s\n' \
    "$PIP_INDEX_URL" "$PIP_TRUSTED_HOST" > /etc/pip.conf
rm -rf /root/.cache/pip
echo SETUP-OK
SCRIPT
)

echo "==> Installing inside the sandbox (~6 min; watch /tmp/setup.log)"
B64=$(printf '%s\n' "$SETUP" | base64 -w0)
# setsid + a log file, not a foreground exec: `aenv exec` would hold the
# connection open for the whole install and a dropped ssh session would take the
# install with it.
aenv exec "$SID" sh -c "echo $B64 | base64 -d > /tmp/setup.sh; \
    sh -c 'setsid nohup sh /tmp/setup.sh > /tmp/setup.log 2>&1 &'"

while : ; do
    sleep 20
    if aenv exec "$SID" sh -c 'grep -q SETUP-OK /tmp/setup.log' 2>/dev/null; then
        echo "    install finished"
        break
    fi
    aenv exec "$SID" sh -c 'tail -1 /tmp/setup.log' 2>/dev/null || true
done

echo "==> What the sandbox ended up with"
aenv exec "$SID" python -c \
    "import openpyxl, reportlab, pdfplumber, docx, pptx, xlsxwriter, pypdf, fitz, sympy, networkx, spellchecker, ms_agent; print('python packages ok')"
aenv exec "$SID" sh -c \
    'for b in ffmpeg convert rg git curl zip unzip 7z jq sqlite3 tree file bc pdftotext; do command -v $b >/dev/null && echo "$b ok" || echo "$b MISSING"; done'

echo "==> Snapshotting as '$NAME'"
aenv exec "$SID" sh -c 'rm -f /tmp/setup.sh /tmp/setup.log'
aenv snapshot create "$SID" --name "$NAME"
aenv delete "$SID" >/dev/null 2>&1 || true

echo
echo "Verify from the training host, which reaches it by the same name:"
echo "    AENV_TEMPLATE=$NAME  # then run the boot check in README.md ('Verify a sandbox boots')"
