#!/bin/bash
set -e

# Copy HF settings (with accounts, no local proxy)
if [ -f /app/settings-hf.json ]; then
    cp /app/settings-hf.json /app/settings.json
fi

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 -ac &
sleep 1

# Start the API server
exec python main.py