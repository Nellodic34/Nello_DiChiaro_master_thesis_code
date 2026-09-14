#!/usr/bin/env bash

# One-window replay of a recorded flight through the live stack.
#
# Plays the bag, runs the triangulation node against it and shows the result in
# the live monitor next to the detection images the drones produced in flight.
# Everything dies together when you hit Ctrl-C, and the run is scored on the way
# out so you can tell whether a change to the filter actually helped.
#
#   ./replay_bag_a.sh                    # look at it
#   ./replay_bag_a.sh --no-images        # same, without the two rqt windows
#   ./replay_bag_a.sh --no-score         # don't record, don't score
#   ./replay_bag_a.sh --start-offset 150 # skip further into the flight
#
# The score is printed and also left in <recording>/score.txt. For the full
# evaluator report - JSON, per-sample CSV and plots under debug/evaluation/ -
# run DRONE_RUN_MODE=decentralized_evaluation ./replay_bag_a.sh; it costs CPU and
# up to a minute of shutdown while it renders, which is why it is not the default.
#
# The second "ground truth" bag is not needed: the triangulation node already
# republishes /target/ground_truth_aligned from the target's own odometry, with
# the profile's z datum correction applied.

set -eo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

BAG="${BAG:-/home/nello/datasets/Ros_bags/temesvar-august/merged/A_17-21_stereo_ok}"
START_OFFSET=105          # the estimate only exists from +107 s of this bag
RATE=1.0
CLOCK_HZ=200
SHOW_IMAGES=1
SCORE=1
REC_DIR="${REC_DIR:-$HOME/replay_runs/$(date +%Y%m%d_%H%M%S)}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bag) BAG="$2"; shift 2 ;;
        --start-offset) START_OFFSET="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        --no-images) SHOW_IMAGES=0; shift ;;
        --no-score) SCORE=0; shift ;;
        -h|--help) sed -n '3,23p' "$0"; exit 0 ;;
        *) echo "Opzione sconosciuta: $1"; exit 1 ;;
    esac
done

[ -d "$BAG" ] || { echo "Bag non trovato: $BAG"; exit 1; }

# ── Environment ──────────────────────────────────────────────────────────────
# Both of these make every sourced setup file echo its own body; a run turns into
# 180 kB of colcon trace with the node's log buried in it.
unset AMENT_TRACE_SETUP_FILES COLCON_TRACE

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
set +eu; source "$ROS_SETUP"; set -eo pipefail
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRIPT_DIR/.venv/bin/activate}"
[ -f "$VENV_ACTIVATE" ] && source "$VENV_ACTIVATE"

# rqt is a system package and the venv is built with include-system-site-packages=false, so
# it has no PyQt5: run under the venv's python it dies on import. Its shebang is
# /usr/bin/env python3, so the only thing that decides is which python3 comes first on PATH.
# Strip the venv's bin instead of saving PATH before the activate above - the caller may
# already have activated the venv, and then there is no clean PATH left to save.
VENV_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin}"
if [ -n "$VENV_BIN" ]; then
    SYS_PATH="${PATH//$VENV_BIN:/}"
    SYS_PATH="${SYS_PATH//:$VENV_BIN/}"
else
    SYS_PATH="$PATH"
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-45}"   # own domain: don't disturb a live session

PIDS=()
RECORD_PID=""

# Each child gets its own process group. The launchers are wrapper scripts that
# spawn python underneath, so killing the wrapper's pid leaves the node running -
# killing the group takes the whole tree.
run_bg() {
    setsid "$@" &
    PIDS+=($!)
}

cleanup() {
    trap - INT TERM EXIT
    echo
    # The recorder writes metadata.yaml from its shutdown path, and without that file
    # the bag cannot be reopened at all. SIGTERM runs it; SIGINT does not - measured,
    # not assumed. Give it time before insisting.
    if [ -n "$RECORD_PID" ]; then
        kill -TERM "$RECORD_PID" 2>/dev/null || true
        for _ in $(seq 20); do
            kill -0 "$RECORD_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL -- "-$RECORD_PID" 2>/dev/null || true
    fi
    kill "${PLAY_PID:-}" 2>/dev/null || true

    # Everything this run starts, matched on this worktree's own paths so a stack
    # running from the main checkout is left alone.
    STACK_PATHS=(nodes/triangulation_node.py nodes/live_monitor.py
                 nodes/evaluate_prediction_node.py launch_triangulation_node.sh
                 launch_live_monitor.sh)

    # The group TERM alone is not enough: the pid bash reports for a backgrounded
    # setsid is not always the group leader, so it can miss the launcher entirely -
    # and then the launcher's own trap never fires, so the evaluator it supervises
    # never hears about the shutdown and just keeps running until it is killed.
    # Signal the paths too.
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for target in "${STACK_PATHS[@]}"; do
        pkill -TERM -f "$SCRIPT_DIR/$target" 2>/dev/null || true
    done

    # The evaluator writes json, then csv, then five plots at dpi 180 from that
    # shutdown path, and it ignores further signals until they are all on disk, so
    # only KILL can stop it. Two seconds of grace left runs holding a
    # prediction_eval.json from the periodic timer and nothing else. Only pay the
    # wait when it is actually running, which by default it is not.
    for _ in $(seq 120); do
        pgrep -f "$SCRIPT_DIR/nodes/evaluate_prediction_node.py" >/dev/null 2>&1 || break
        sleep 0.5
    done
    sleep 2

    # A node that outlives its run becomes a second publisher on
    # /target/odometry_estimate in the NEXT run - which is what made the monitor
    # look like it was teleporting.
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    for target in "${STACK_PATHS[@]}"; do
        pkill -KILL -f "$SCRIPT_DIR/$target" 2>/dev/null || true
    done

    if [ "$SCORE" = "1" ] && [ -d "$REC_DIR" ]; then
        echo "=== punteggio della corsa ==="
        # Keep it next to the bag it describes: the terminal scrolls, the file does not.
        # stdout only: numpy's warnings belong on the terminal, not in the file.
        python3 "$SCRIPT_DIR/dataset_tools/score_replay.py" "$REC_DIR" \
            | tee "$REC_DIR/score.txt" || true
        echo "registrazione: $REC_DIR   (punteggio anche in score.txt)"
    fi
}
trap cleanup INT TERM EXIT

# ── Zenoh router: without it no node sees any other ──────────────────────────
if pgrep -f rmw_zenohd >/dev/null; then
    echo "router zenoh: gia' attivo"
else
    echo "router zenoh: lo avvio"
    run_bg ros2 run rmw_zenoh_cpp rmw_zenohd
    sleep 4
fi

# ── The stack under test ─────────────────────────────────────────────────────
# Sim time is not optional here: without it the node compares the bag's stamps
# against the wallclock and discards nearly every observation as too old.
#
# Plain 'decentralized', not the configured 'decentralized_evaluation': the
# evaluator node adds nothing here - the node publishes /target/predicted_trajectory
# itself - while it burns CPU writing PNG and CSV, and it holds the launcher open
# for seconds on shutdown, which is how runs kept leaking a live node.
echo "nodo di triangolazione (sim time, profilo temesvar_A)"
run_bg env USE_SIM_TIME=true BAG_PROFILE=temesvar_A \
    DRONE_RUN_MODE="${DRONE_RUN_MODE:-decentralized}" \
    "$SCRIPT_DIR/launch_triangulation_node.sh"

echo "live monitor"
run_bg "$SCRIPT_DIR/launch_live_monitor.sh"

if [ "$SCORE" = "1" ]; then
    mkdir -p "$(dirname "$REC_DIR")"
    setsid ros2 bag record -s mcap -o "$REC_DIR" \
        /target/odometry_estimate /target/triangulation_raw /target/ground_truth_aligned \
        >/dev/null 2>&1 &
    RECORD_PID=$!
fi

echo "attendo che i nodi si registrino..."
sleep 8

# ── Playback ─────────────────────────────────────────────────────────────────
# The bag carries the estimate the stack produced during the flight itself. Left
# in, rosbag2_player becomes a second publisher on /target/odometry_estimate and
# the monitor draws both streams at once - and since the recorded one is in the
# unaligned datum, ~8 m higher, the trail jumps between the two every frame.
#
# Not -x: that flag is a regex matched against the WHOLE topic name, so the
# obvious '^/target/' matches nothing at all and silently excludes nothing.
# --exclude-topics takes plain names and cannot be misread.
echo "riproduco $BAG da +${START_OFFSET}s  (Ctrl-C per fermare e vedere il punteggio)"
ros2 bag play "$BAG" \
    --clock "$CLOCK_HZ" --start-offset "$START_OFFSET" --rate "$RATE" \
    --exclude-topics \
        /target/odometry_estimate /target/pose_estimate /target/position_estimate \
        /target/position_error /target/predicted_trajectory /target/triangulation_raw \
        /target/triangulation_evaluation /target/ground_truth_aligned \
        /tracker/fusion_mode /uav11/triangulation_debug_annotated/compressed &
PLAY_PID=$!

if [ "$SHOW_IMAGES" = "1" ]; then
    # rqt_image_view resolves the message type from the live graph when it starts.
    # Started before the player, it finds no such topic, subscribes to nothing and
    # shows an empty dropdown - which is what it did when it was started up front.
    sleep 6
    for topic in /uav11/detection2/compressed /uav8/detection2/compressed; do
        run_bg env -u VIRTUAL_ENV PATH="$SYS_PATH" \
            ros2 run rqt_image_view rqt_image_view "$topic"
    done
fi

wait "$PLAY_PID"
echo "bag finito."
