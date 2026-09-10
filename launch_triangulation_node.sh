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
except Exception:
    pass
"
}

OBSERVER1="${OBSERVER1_NAME:-$(get_yaml_value "uav_roles.observer1")}"
OBSERVER2="${OBSERVER2_NAME:-$(get_yaml_value "uav_roles.observer2")}"
TARGET="${TARGET_NAME:-$(get_yaml_value "uav_roles.target")}"
USE_SIM_TIME="${USE_SIM_TIME:-$(get_yaml_value "launcher.use_sim_time")}" 
RUN_MODE="${DRONE_RUN_MODE:-$(get_yaml_value "launcher.drone_run_mode")}"
KF_ACC="${KF_PROCESS_NOISE_ACC:-$(get_yaml_value "ekf.kf_process_noise_acc")}"
KF_MEAS="${KF_MEASUREMENT_NOISE:-$(get_yaml_value "ekf.kf_measurement_noise")}"
KF_INIT="${KF_INITIAL_COVARIANCE:-$(get_yaml_value "ekf.kf_initial_covariance")}"
KF_VEL="${KF_PROCESS_NOISE_VEL:-$(get_yaml_value "ekf.kf_process_noise_vel")}"
MAX_PAIR_AGE_SEC="${MAX_PAIR_AGE_SEC:-$(get_yaml_value "triangulation.max_pair_age_sec")}"
OBSERVATION_BUFFER_SEC="${OBSERVATION_BUFFER_SEC:-$(get_yaml_value "triangulation.observation_buffer_sec")}"
TRIANGULATION_MIN_BASELINE_M="${TRIANGULATION_MIN_BASELINE_M:-$(get_yaml_value "triangulation.triangulation_min_baseline_m")}"
MAX_TRIANGULATION_ERROR_M="${MAX_TRIANGULATION_ERROR_M:-$(get_yaml_value "triangulation.max_triangulation_error_m")}"
PROCESSING_RATE_HZ="${PROCESSING_RATE_HZ:-$(get_yaml_value "triangulation.processing_rate_hz")}"
MAX_COAST_SEC="${MAX_COAST_SEC:-$(get_yaml_value "triangulation.max_coast_sec")}"

if [ -z "$USE_SIM_TIME" ]; then
    USE_SIM_TIME="false"
fi

if [ -z "$RUN_MODE" ]; then
    RUN_MODE="decentralized"
fi

if [ -z "$OBSERVER1" ] || [ -z "$OBSERVER2" ] || [ -z "$TARGET" ]; then
    echo "Could not resolve observer or target names from runtime_stack.yaml."
    exit 1
fi

echo "==========================================="
echo "Starting Triangulation Node"
echo "Observer 1:   ${OBSERVER1}"
echo "Observer 2:   ${OBSERVER2}"
echo "Target:       ${TARGET}"
echo "Run Mode:     ${RUN_MODE}"
echo "Sim Time:     ${USE_SIM_TIME}"
echo "==========================================="

start_triangulation_node() {
    python3 "$SCRIPT_DIR/nodes/triangulation_node.py" \
        --ros-args \
        -p observer1_name:="$OBSERVER1" \
        -p observer2_name:="$OBSERVER2" \
        -p target_name:="$TARGET" \
        -p use_sim_time:="$USE_SIM_TIME" \
        -p kf_process_noise_acc:="$KF_ACC" \
        -p kf_measurement_noise:="$KF_MEAS" \
        -p kf_initial_covariance:="$KF_INIT" \
        -p kf_process_noise_vel:="$KF_VEL" \
        -p max_pair_age_sec:="$MAX_PAIR_AGE_SEC" \
        -p observation_buffer_sec:="$OBSERVATION_BUFFER_SEC" \
        -p triangulation_min_baseline_m:="$TRIANGULATION_MIN_BASELINE_M" \
        -p max_triangulation_error_m:="$MAX_TRIANGULATION_ERROR_M" \
        -p processing_rate_hz:="$PROCESSING_RATE_HZ" \
        -p max_coast_sec:="$MAX_COAST_SEC" \
        "$@"
}

if [ "$RUN_MODE" = "decentralized_evaluation" ]; then
    cleanup() {
        kill "${TRIANGULATION_PID:-}" 2>/dev/null || true
        kill "${EVALUATION_PID:-}" 2>/dev/null || true
        # The evaluator writes its plots and report from its shutdown path, which takes seconds.
        # Returning here without waiting is what truncated every run to json+csv.
        wait "${EVALUATION_PID:-}" 2>/dev/null || true
    }

    trap cleanup INT TERM EXIT

    start_triangulation_node "$@" &
    TRIANGULATION_PID=$!

    echo "Starting Prediction Evaluator Node"
    python3 "$SCRIPT_DIR/nodes/evaluate_prediction_node.py" \
        --ros-args \
        -p use_sim_time:="$USE_SIM_TIME" &
    EVALUATION_PID=$!

    if wait "$TRIANGULATION_PID"; then
        TRIANGULATION_STATUS=0
    else
        TRIANGULATION_STATUS=$?
    fi

    kill "$EVALUATION_PID" 2>/dev/null || true
    wait "$EVALUATION_PID" 2>/dev/null || true
    trap - INT TERM EXIT
    exit "$TRIANGULATION_STATUS"
fi

start_triangulation_node "$@"