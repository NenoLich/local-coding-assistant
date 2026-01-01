#!/bin/sh
set -e

# Defaults to /workspace/ipc if not set
IPC_BASE_DIR=${IPC_DIR:-/workspace/ipc}

# Check if we are using a nested session path
IPC_PARENT=$(dirname "$IPC_BASE_DIR")

# Ensure the parent directory exists
if [ ! -d "$IPC_PARENT" ]; then
    mkdir -p "$IPC_PARENT"
fi

# Ensure locca can write to the parent directory to allow deletion of the session dir
# We only do this check to avoid errors, but we apply ownership/permissions if possible
if [ -d "$IPC_PARENT" ]; then
    chown locca:locca "$IPC_PARENT" || true
    chmod 775 "$IPC_PARENT" || true
fi

# Create IPC subdirectories unconditionally
mkdir -p "$IPC_BASE_DIR/requests" "$IPC_BASE_DIR/responses"

# Set ownership and permissions for IPC directory structure
chown -R locca:locca "$IPC_BASE_DIR"
find "$IPC_BASE_DIR" -type d -exec chmod 750 {} +
find "$IPC_BASE_DIR" -type f -exec chmod 640 {} +

# Ensure the requests and responses directories have the right permissions
chmod 750 "$IPC_BASE_DIR/requests" "$IPC_BASE_DIR/responses"

# Execute the command as the locca user
exec gosu locca "$@"
