#!/usr/bin/env bash

# Launch the live monitor (matplotlib visualization) on a PC.
# This script sets up the ROS 2 environment and virtual environment, then launches the node.

set -eo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [ -f "$ROS_SETUP" ]; then
  set +eu
  source "$ROS_SETUP"
  set -eo pipefail
elif ls /opt/ros/*/setup.bash 1>/dev/null 2>&1; then
  set +eu
  source /opt/ros/*/setup.bash
  set -eo pipefail
else
  echo "WARNING: No ROS 2 installation found in /opt/ros/"
fi

ROS2_WS_LOCAL_SETUP="${ROS2_WS_LOCAL_SETUP:-$HOME/ros2_ws/install/local_setup.bash}"
if [ -z "${LOCAL_SETUP:-}" ]; then
  if [ -f "$ROS2_WS_LOCAL_SETUP" ]; then
    LOCAL_SETUP="$ROS2_WS_LOCAL_SETUP"
  fi
fi

if [ -f "$LOCAL_SETUP" ]; then
  set +eu
  source "$LOCAL_SETUP"
  set -eo pipefail
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRIPT_DIR/.venv/bin/activate}"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
LIVE_MONITOR_PY="$SCRIPT_DIR/nodes/live_monitor.py"

if [ ! -f "$LIVE_MONITOR_PY" ]; then
  echo "Error: live monitor python script not found: $LIVE_MONITOR_PY"
  exit 1
fi

echo "=== Launching Live Monitor ==="
exec "$PYTHON_BIN" "$LIVE_MONITOR_PY" "$@"

