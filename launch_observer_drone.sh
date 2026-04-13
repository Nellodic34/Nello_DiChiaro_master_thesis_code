#!/bin/bash
# Launch the observer drone node (yolov8_observer_node.py)
# Run this on the OBSERVER drone (observer2 from config).
# The observer publishes ray + camera_origin for the main drone to consume.

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

# Source virtual environment: local .venv → workspace venv → system python
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$WORKSPACE_VENV" ]]; then
    echo "INFO: Using workspace venv: $WORKSPACE_VENV"
    source "$WORKSPACE_VENV"
else
    echo "WARNING: No virtualenv found. Run 'bash setup_env.sh' first, or pip install pyyaml ultralytics"
fi
set -eo pipefail

# Read config
CONFIG="$SCRIPT_DIR/config/runtime_stack.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

OBSERVER1=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['uav_roles']['observer1'])")
OBSERVER2=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['uav_roles']['observer2'])")
TARGET=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['uav_roles']['target'])")
MODEL_PATH="${1:-$SCRIPT_DIR/weights/best_openvino_model}"

echo "=== Observer Drone Node ==="
echo "Running on: $OBSERVER2 (observer2)"
echo "Target:     $TARGET"
echo "Model:      $MODEL_PATH"
echo "=========================="

python3 "$SCRIPT_DIR/nodes/yolov8_observer_node.py" \
    --ros-args \
    -p uav_name:="$OBSERVER2" \
    -p target_name:="$TARGET" \
    -p model_path:="$MODEL_PATH" \
    -p imgsz:=896
