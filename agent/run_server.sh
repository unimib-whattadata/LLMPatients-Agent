#!/bin/bash
export PYTHONPATH=$(pwd)
uvicorn agent.api.app:app --host 0.0.0.0 --port 8000 --reload