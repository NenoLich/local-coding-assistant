FROM python:3.12-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with known UID/GID
RUN groupadd -g 1000 locca && \
    useradd -m -u 1000 -g locca locca

# Set working directory
WORKDIR /workspace
RUN chown locca:locca /workspace

# Create IPC directory structure under workspace
RUN mkdir -p /workspace/ipc && \
    chown -R locca:locca /workspace/ipc

# Copy and set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Install python dependencies
RUN pip install --no-cache-dir pyyaml pydantic python-dotenv psutil

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Set entrypoint and default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
