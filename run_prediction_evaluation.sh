#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
RUNTIME_CONFIG="$SCRIPT_DIR/config/runtime_stack.yaml"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ROS2_WS_LOCAL_SETUP="${ROS2_WS_LOCAL_SETUP:-$HOME/ros2_ws/install/local_setup.bash}"
if [ -z "${LOCAL_SETUP:-}" ] && [ -f "$ROS2_WS_LOCAL_SETUP" ]; then
  LOCAL_SETUP="$ROS2_WS_LOCAL_SETUP"
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRIPT_DIR/.venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_SCRIPT="$SCRIPT_DIR/nodes/evaluate_prediction_node.py"
AUTO_OPEN_RESULTS="${AUTO_OPEN_RESULTS:-0}"

get_yaml_value() {
  local key_path="$1"
  python3 -c "
import yaml
try:
    config = yaml.safe_load(open('$RUNTIME_CONFIG'))
    val = config
    for key in '$key_path'.split('.'):
        val = val[key]
    print(val)
except Exception:
    pass
"
}

USE_SIM_TIME="${USE_SIM_TIME:-$(get_yaml_value "launcher.use_sim_time")}" 
if [ -z "$USE_SIM_TIME" ]; then
  USE_SIM_TIME="false"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/debug/evaluation}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

JSON_PATH="$RUN_DIR/prediction_eval.json"
PLOT_PATH="$RUN_DIR/prediction_eval_plot.png"
XY_PLOT_PATH="$RUN_DIR/prediction_eval_xy.png"
REPORT_PATH="$RUN_DIR/prediction_eval_report.md"
CSV_PATH="$RUN_DIR/prediction_eval_samples.csv"

ln -sfn "$RUN_DIR" "$OUTPUT_DIR/latest"

echo "[Evaluator] output directory: $RUN_DIR"
echo "[Evaluator] stop with Ctrl+C to finalize plot/report"

if [ -f "$ROS_SETUP" ]; then
  set +eu
  source "$ROS_SETUP"
  set -euo pipefail
elif ls /opt/ros/*/setup.bash 1>/dev/null 2>&1; then
  set +eu
  source /opt/ros/*/setup.bash
  set -euo pipefail
fi

if [ -n "${LOCAL_SETUP:-}" ] && [ -f "$LOCAL_SETUP" ]; then
  set +eu
  source "$LOCAL_SETUP"
  set -euo pipefail
fi

if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
fi

"$PYTHON_BIN" "$EVAL_SCRIPT" --ros-args \
  -p use_sim_time:="$USE_SIM_TIME" \
  -p output_dir:="$RUN_DIR" \
  -p output_json_path:="$JSON_PATH" \
  -p output_plot_path:="$PLOT_PATH" \
  -p output_xy_plot_path:="$XY_PLOT_PATH" \
  -p output_report_path:="$REPORT_PATH" \
  -p output_csv_path:="$CSV_PATH" \
  "$@"

echo ""
echo "[Evaluator] artifacts:"
echo "  - $JSON_PATH"
echo "  - $PLOT_PATH"
echo "  - $XY_PLOT_PATH"
echo "  - $REPORT_PATH"
echo "  - $CSV_PATH"

if [[ "$AUTO_OPEN_RESULTS" == "1" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$PLOT_PATH" >/dev/null 2>&1 || true
  xdg-open "$XY_PLOT_PATH" >/dev/null 2>&1 || true
  xdg-open "$REPORT_PATH" >/dev/null 2>&1 || true
fi
