#!/usr/bin/env bash
# Source before running anything in this project:  source tools/env.sh
# CUDA: the system libcuda does not match the 580.173.02 kernel module (error 804 /
# "Driver/library version mismatch"). Dynamically link the matching user-space
# driver libraries first, as ESAM/local/env.sh does.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NVIDIA_DRIVER_LIBS="${NVIDIA_DRIVER_LIBS:-/home/yqx/.cache/openjev/driver-580.173.02}"
export LD_LIBRARY_PATH="$NVIDIA_DRIVER_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLBACKEND=Agg
export YOLO_CONFIG_DIR="$PROJECT_ROOT/work_dirs/.yolo_config"
source /home/yqx/SegmentAnything3D/AutoSeg3D/.venv/bin/activate
