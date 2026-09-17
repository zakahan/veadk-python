# Case-owned Dockerfile: deploy.py bundles the pinned source from local Git.
FROM agentkit-prod-public-cn-beijing.cr.volces.com/base/py-simple:python3.12-bookworm-slim-latest

ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
COPY vendor/veadk.zip /tmp/veadk.zip
RUN python - <<'PY'
from pathlib import Path
import re

requirements = Path("requirements.txt").read_text()
Path("/tmp/build-requirements.txt").write_text(
    re.sub(r"https://github.com/volcengine/veadk-python/archive/[0-9a-f]{40}\.zip",
           "file:///tmp/veadk.zip", requirements)
)
PY
RUN uv pip install -r /tmp/build-requirements.txt && rm /tmp/veadk.zip /tmp/build-requirements.txt
COPY . .
RUN rm -rf vendor

EXPOSE 8000
CMD ["python", "app.py"]
