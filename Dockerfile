### Not necessarily working

# Use a specific platform if building on Apple Silicon for a Linux server
FROM --platform=linux/arm64 python:3.12-slim-bookworm

# Install uv binaries from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and use copy mode for linking
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for better caching. The relabel is important for podman on macos.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,relabel=shared \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,relabel=shared \
    uv sync --frozen --no-install-project --no-dev

COPY ./ ./

# Final sync to install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# seems goofy
WORKDIR /app/app

# Run the training with the defaults
CMD ["uv", "run", "python", "train.py"]