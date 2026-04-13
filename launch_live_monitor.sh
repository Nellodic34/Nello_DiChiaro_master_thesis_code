#!/bin/bash
# Launch the live monitor (matplotlib visualization) on a PC.
# This does NOT need the virtual environment — only ROS 2 + matplotlib.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_VENV="${HOME}/git/training/mrs_uav_flightforge_simulator/.venv/bin/activate"

# Source ROS 2 (set +eu: ROS setup scripts have unbound variables)
set +eu
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
elif ls /opt/ros/*/setup.bash 1>/dev/null 2>&1; then
    source /opt/ros/*/setup.bash
else
    echo "ERROR: No ROS 2 installation found in /opt/ros/"; exit 1
fi

# Source virtual environment (needed for pyyaml config parsing)
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$WORKSPACE_VENV" ]]; then
    source "$WORKSPACE_VENV"
fi
set -eo pipefail

# Read config
CONFIG="$SCRIPT_DIR/config/runtime_stack.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

TARGET=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['uav_roles']['target'])")

echo "=== Live Monitor ==="
echo "Target: $TARGET"
echo "===================="

python3 "$SCRIPT_DIR/nodes/live_monitor.py" \
    --ros-args \
    -p target_name:="$TARGET"
