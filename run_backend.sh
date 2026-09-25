#!/usr/bin/env bash
# Start the FastAPI backend. Run from the project root: bash run_backend.sh
set -e
uvicorn backend.main:app --reload --port 8000
