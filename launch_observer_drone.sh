#!/usr/bin/env bash

set -eo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
RUNTIME_CONFIG="$SCRIPT_DIR/config/runtime_stack.yaml"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ROS2_WS_LOCAL_SETUP="${ROS2_WS_LOCAL_SETUP:-$HOME/ros2_ws/install/local_setup.bash}"

if [ -z "${LOCAL_SETUP:-}" ] && [ -f "$ROS2_WS_LOCAL_SETUP" ]; then
    LOCAL_SETUP="$ROS2_WS_LOCAL_SETUP"
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRIPT_DIR/.venv/bin/activate}"

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

if [ -n "${LOCAL_SETUP:-}" ] && [ -f "$LOCAL_SETUP" ]; then
    set +eu
    source "$LOCAL_SETUP"
    set -eo pipefail
fi

if [ -f "$VENV_ACTIVATE" ]; then source "$VENV_ACTIVATE"; fi

get_yaml_value() {
    local key_path="$1"
    python3 -c "
import yaml
try:
    config = yaml.safe_load(open('$RUNTIME_CONFIG'))
    keys = '$key_path'.split('.')
    val = config
    for k in keys:
        val = val[k]
    print(val)
except:
    pass
"
}

DRONE_RUN_MODE=$(get_yaml_value "launcher.drone_run_mode")
USE_SIM_TIME="${USE_SIM_TIME:-$(get_yaml_value "launcher.use_sim_time")}" 
OBSERVER1=$(get_yaml_value "uav_roles.observer1")
OBSERVER2=$(get_yaml_value "uav_roles.observer2")
MODEL_PATH_RAW="${MODEL_PATH:-$(get_yaml_value "vision.model_path")}" 
LOCAL_UAV="${UAV_NAME:-}"
LOCAL_OBSERVER_ROLE="${OBSERVER_ROLE:-}"

if [ -z "$MODEL_PATH_RAW" ]; then
    MODEL_PATH_RAW="dataset_tools/weights/best_openvino_model"
fi

if [ "${MODEL_PATH_RAW#/}" = "$MODEL_PATH_RAW" ]; then
    MODEL_PATH="$SCRIPT_DIR/$MODEL_PATH_RAW"
else
    MODEL_PATH="$MODEL_PATH_RAW"
fi

if [ -z "$USE_SIM_TIME" ]; then
    USE_SIM_TIME="false"
fi

case "$LOCAL_OBSERVER_ROLE" in
    ""|observer1|observer2)
        ;;
    *)
        echo "Invalid OBSERVER_ROLE='$LOCAL_OBSERVER_ROLE' (expected observer1 or observer2)"
        exit 1
        ;;
esac

if [ -z "$LOCAL_OBSERVER_ROLE" ] && [ -n "$LOCAL_UAV" ]; then
    if [ "$LOCAL_UAV" = "$OBSERVER1" ] && [ "$LOCAL_UAV" = "$OBSERVER2" ] && [ -n "$LOCAL_UAV" ]; then
        echo "UAV_NAME='$LOCAL_UAV' matches both observer1 and observer2 in runtime_stack.yaml; set OBSERVER_ROLE explicitly."
        exit 1
    elif [ "$LOCAL_UAV" = "$OBSERVER1" ]; then
        LOCAL_OBSERVER_ROLE="observer1"
    elif [ "$LOCAL_UAV" = "$OBSERVER2" ]; then
        LOCAL_OBSERVER_ROLE="observer2"
    fi
fi

if [ -z "$LOCAL_OBSERVER_ROLE" ]; then
    LOCAL_OBSERVER_ROLE="observer2"
fi

if [ -z "$LOCAL_UAV" ]; then
    LOCAL_UAV=$(get_yaml_value "uav_roles.${LOCAL_OBSERVER_ROLE}")
fi

if [ -z "$LOCAL_UAV" ]; then
    echo "Could not resolve local UAV. Set UAV_NAME or OBSERVER_ROLE, or fill uav_roles.${LOCAL_OBSERVER_ROLE} in runtime_stack.yaml."
    exit 1
fi

echo "==========================================="
echo "Starting OBSERVER Drone Vision Stack (Eye)"
echo "Run Mode:     ${DRONE_RUN_MODE:-unknown}"
echo "Sim Time:     ${USE_SIM_TIME}"
echo "Role:         ${LOCAL_OBSERVER_ROLE}"
echo "Local UAV:    ${LOCAL_UAV}"
echo "==========================================="

PIDS=""
trap "echo 'Stopping all nodes...'; kill $PIDS 2>/dev/null; exit" SIGINT SIGTERM

if [ "$DRONE_RUN_MODE" = "simple_detection" ]; then
    echo "Starting Simple Detection Node (Local)..."
    python3 "$SCRIPT_DIR/nodes/test_yolov8_realtime_node.py" \
        --ros-args -p uav_name:="$LOCAL_UAV" -p use_sim_time:="$USE_SIM_TIME" -p model_path:="$MODEL_PATH" "$@" &
    PIDS="$PIDS $!"

elif [[ "$DRONE_RUN_MODE" == "decentralized"* ]]; then
    echo "Starting Decentralized Observer Node..."
    python3 "$SCRIPT_DIR/nodes/observer_node.py" \
        --ros-args -p uav_name:="$LOCAL_UAV" -p observer_role:="$LOCAL_OBSERVER_ROLE" -p use_sim_time:="$USE_SIM_TIME" -p model_path:="$MODEL_PATH" "$@" &
    PIDS="$PIDS $!"

else
    echo "Unknown or missing launcher.drone_run_mode in runtime_stack.yaml: $DRONE_RUN_MODE"
    exit 1
fi

echo "All requested nodes are running in the background."
echo "Press CTRL+C to stop."
wait
