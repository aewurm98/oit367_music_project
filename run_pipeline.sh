#!/bin/bash
# OIT367 Pipeline Runner
# Run this from your terminal: bash run_pipeline.sh
python3 /tmp/run_models.py > /sessions/sharp-awesome-hopper/mnt/OIT-367/model_run_log.txt 2>&1
echo "Exit code: $?" >> /sessions/sharp-awesome-hopper/mnt/OIT-367/model_run_log.txt
