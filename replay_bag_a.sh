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
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done

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
echo "nodo di triangolazione (sim time, profilo temesvar_A)"
run_bg env USE_SIM_TIME=true BAG_PROFILE=temesvar_A "$SCRIPT_DIR/launch_triangulation_node.sh"

echo "live monitor"
run_bg "$SCRIPT_DIR/launch_live_monitor.sh"

if [ "$SHOW_IMAGES" = "1" ]; then
    # The node's own ray overlay only draws on observer1 (uav11) and this bag has no
    # uav11 colour image, so it would stay blank. These two are what the detectors
    # actually saw in flight, boxes included.
    for topic in /uav11/detection2/compressed /uav8/detection2/compressed; do
        run_bg ros2 run rqt_image_view rqt_image_view "$topic"
    done
fi

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
# Drop the /target and /tracker topics the previous flight left in the bag: they
# would sit on top of what the node is publishing right now and the monitor would
# plot the old run instead of this one.
echo "riproduco $BAG da +${START_OFFSET}s  (Ctrl-C per fermare e vedere il punteggio)"
ros2 bag play "$BAG" \
    --clock "$CLOCK_HZ" --start-offset "$START_OFFSET" --rate "$RATE" \
    -x '^/target/|^/tracker/'

echo "bag finito."
