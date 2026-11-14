# === Build stage ===
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (frozen lockfile, no editable install, no dev extras)
RUN uv sync --frozen --no-dev --no-editable

# === Runtime stage ===
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-privileged system user for runtime security
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -s /bin/sh -m appuser

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Ensure venv is in PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
