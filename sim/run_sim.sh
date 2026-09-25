#!/usr/bin/env bash
# Start the FlightForge sim with THIS repo's nodes in the perception panes.
#
#   ./sim/run_sim.sh
#
# Everything sim-side stays in the sim tree; the only edit it needs is
# launch_sim.patch, which turns three hardcoded pane paths into env-defaulted ones,
# exactly like the two next to them already are.
set -eo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SIM_TREE="${SIM_TREE:-$HOME/git/training/mrs_uav_flightforge_simulator/tmux/one_drone}"

if ! grep -q 'TRIANGULATION_PANE_SCRIPT' "$SIM_TREE/launch_sim.sh"; then
    # Printed as a single cwd-independent command on purpose: the 'cd X && patch ...'
    # form gets pasted as one line and then cd swallows the rest as arguments.
    echo "launch_sim.sh non e' ancora patchato. Applica prima:"
    echo "  patch -d $SIM_TREE launch_sim.sh < $HERE/launch_sim.patch"
    exit 1
fi

cd "$SIM_TREE"
exec env \
    OBSERVER_PANE_SCRIPT="$HERE/observer2.sh" \
    MAIN_PANE_SCRIPT="$HERE/observer1.sh" \
    TRIANGULATION_PANE_SCRIPT="$HERE/triangulation.sh" \
    LIVE_MONITOR_SCRIPT="$HERE/live_monitor.sh" \
    PREDICTION_EVALUATION_SCRIPT="$HERE/evaluation.sh" \
    ./launch_sim.sh "$@"
