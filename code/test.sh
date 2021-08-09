#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
export PYTHONPATH=./:$PYTHONPATH




python -u test.py \
        2>&1

