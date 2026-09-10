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
        -h|--help) sed -n '3,17p' "$0"; exit 0 ;;
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
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done

    # And then check, because none of the above is reliable on its own: a launcher
    # that traps TERM sits in wait() while its children shut down, and the pid bash
    # reports for a backgrounded setsid is not always the group leader, so the group
    # kill can miss. A node that outlives its run becomes a second publisher on
    # /target/odometry_estimate in the NEXT run - which is what made the monitor
    # look like it was teleporting. Matched on this worktree's own paths, so a
    # stack running from the main checkout is left alone.
    for target in nodes/triangulation_node.py nodes/live_monitor.py \
                  nodes/evaluate_prediction_node.py launch_triangulation_node.sh \
                  launch_live_monitor.sh; do
        pkill -KILL -f "$SCRIPT_DIR/$target" 2>/dev/null || true
    done

    if [ "$SCORE" = "1" ] && [ -d "$REC_DIR" ]; then
        echo "=== punteggio della corsa ==="
        python3 "$SCRIPT_DIR/dataset_tools/score_replay.py" "$REC_DIR" || true
        echo "registrazione: $REC_DIR"
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
        run_bg ros2 run rqt_image_view rqt_image_view "$topic"
    done
fi

wait "$PLAY_PID"
echo "bag finito."
