#!/bin/bash
# Launch the main drone node (yolov8_main_triangulation_node.py)
# Run this on the MAIN drone (observer1 from config).
# The main drone uses its own camera + subscribes to remote rays from observer2.
# Optionally also launches evaluate_prediction_node.py if mode is decentralized_evaluation.

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
RUN_MODE=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['launcher']['drone_run_mode'])")
MODEL_PATH="${1:-$SCRIPT_DIR/weights/best_openvino_model}"

echo "=== Main Drone Node ==="
echo "Running on:      $OBSERVER1 (observer1/main)"
echo "Remote observer:  $OBSERVER2 (observer2)"
echo "Target:           $TARGET"
echo "Run mode:         $RUN_MODE"
echo "Model:            $MODEL_PATH"
echo "========================"

# Launch main triangulation node
# uav_name = observer1 (this drone's camera)
# observer2_name = observer2 (remote observer, publishes rays)
python3 "$SCRIPT_DIR/nodes/yolov8_main_triangulation_node.py" \
    --ros-args \
    -p uav_name:="$OBSERVER1" \
    -p observer1_name:="$OBSERVER2" \
    -p observer2_name:="$OBSERVER2" \
    -p target_name:="$TARGET" \
    -p model_path:="$MODEL_PATH" &

MAIN_PID=$!

# If decentralized_evaluation, also launch evaluator
if [[ "$RUN_MODE" == "decentralized_evaluation" ]]; then
    echo "Starting evaluate_prediction_node..."
    sleep 2
    python3 "$SCRIPT_DIR/nodes/evaluate_prediction_node.py" \
        --ros-args \
        -p target_name:="$TARGET" &
    EVAL_PID=$!
fi

# Wait for main process
trap "kill $MAIN_PID ${EVAL_PID:-} 2>/dev/null; exit" INT TERM
wait $MAIN_PID
