FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

ENV UV_COMPILE_BYTECODE=1

RUN --mount=type=cache,target=/root/.cache/uv \
uv pip install --system hatchling

RUN --mount=type=cache,target=/root/.cache/uv \
uv pip install --system -r pyproject.toml

COPY web/ ./web/
COPY data/ ./data/
COPY models/ ./models/

RUN apt-get update -y && apt-get install curl --no-install-recommends -y && \
    curl -o ./models/abussive_detection.keras https://storage.googleapis.com/tutortoise-bucket/model/abussive_detection.keras

RUN --mount=type=cache,target=/root/.cache/uv \
uv pip install --system -e .

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/granian /usr/local/bin/

COPY --from=builder /app/data /app/data
COPY --from=builder /app/models /app/models
COPY --from=builder /app/web /app/web

ENV GRANIAN_HOST=0.0.0.0
ENV GRANIAN_PORT=8000
ENV GRANIAN_INTERFACE=asgi
ENV GRANIAN_WORKERS_PER_CORE=2
ENV GRANIAN_MAX_WORKERS=4
ENV GRANIAN_MIN_WORKERS=2
ENV GRANIAN_HTTP=auto
ENV GRANIAN_BACKLOG=1024
ENV GRANIAN_LOG_LEVEL=info
ENV GRANIAN_LOG_ACCESS_ENABLED=true
ENV GRANIAN_THREADING_MODE=workers
ENV GRANIAN_LOOP=auto
ENV GRANIAN_HTTP1_KEEP_ALIVE=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN echo '#!/usr/bin/env python3\n\
import multiprocessing\n\
import os\n\
\n\
# Get configuration from environment variables\n\
workers_per_core = int(os.getenv("GRANIAN_WORKERS_PER_CORE", 2))\n\
max_workers = int(os.getenv("GRANIAN_MAX_WORKERS", 32))\n\
min_workers = int(os.getenv("GRANIAN_MIN_WORKERS", 2))\n\
\n\
# Calculate workers based on CPU cores\n\
workers = multiprocessing.cpu_count() * workers_per_core\n\
\n\
# Apply limits\n\
workers = min(workers, max_workers)\n\
workers = max(workers, min_workers)\n\
\n\
print(workers)' > /app/calculate_workers.py && chmod +x /app/calculate_workers.py

EXPOSE $GRANIAN_PORT

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
CMD curl -f http://localhost:${GRANIAN_PORT}/health || exit 1

RUN echo '#!/bin/bash\n\
\n\
# Calculate workers\n\
WORKERS=$(python /app/calculate_workers.py)\n\
\n\
echo "Starting Granian with configuration:"\n\
echo "Host: $GRANIAN_HOST"\n\
echo "Port: $GRANIAN_PORT"\n\
echo "Workers: $WORKERS"\n\
echo "Interface: $GRANIAN_INTERFACE"\n\
echo "HTTP Version: $GRANIAN_HTTP"\n\
echo "Threading Mode: $GRANIAN_THREADING_MODE"\n\
echo "Event Loop: $GRANIAN_LOOP"\n\
echo "Log Level: $GRANIAN_LOG_LEVEL"\n\
\n\
exec granian "web.main:app" \\\n\
--host "$GRANIAN_HOST" \\\n\
--port "$GRANIAN_PORT" \\\n\
--interface "$GRANIAN_INTERFACE" \\\n\
--workers "$WORKERS" \\\n\
--http "$GRANIAN_HTTP" \\\n\
--threading-mode "$GRANIAN_THREADING_MODE" \\\n\
--loop "$GRANIAN_LOOP" \\\n\
--backlog "$GRANIAN_BACKLOG" \\\n\
--log-level "$GRANIAN_LOG_LEVEL" \\\n\
$([ "$GRANIAN_LOG_ACCESS_ENABLED" = "true" ] && echo "--access-log") \\\n\
$([ "$GRANIAN_HTTP1_KEEP_ALIVE" = "true" ] && echo "--http1-keep-alive") \\\n\
"$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
