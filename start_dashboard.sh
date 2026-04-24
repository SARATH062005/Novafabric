#!/bin/bash

# Configuration
PROJECT_DIR="/home/sarath/lerobot"
APP_DIR="$PROJECT_DIR/novafab_web"
CONDA_BASE="/home/sarath/miniforge3"

echo "🚀 Starting NOVAFAB Dashboard..."

# Check if port 5000 is already in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port 5000 is already in use. Attempting to clear..."
    kill -9 $(lsof -t -i :5000)
    sleep 1
fi

# Activate environment and run flask
source "$CONDA_BASE/bin/activate"
conda activate lerobot

cd "$APP_DIR"
echo "✅ Dashboard is launching at http://localhost:5000"
python app.py
