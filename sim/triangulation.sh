#!/usr/bin/env bash
# Sim triangulation node, from THIS repo.
#
# No remaps needed: it builds every topic from the observer/target names, which are
# plain env knobs. BAG_PROFILE stays "none" - the per-flight bias constants in
# bag_profiles describe the Temesvar bag's boot tilt, which does not exist in sim.
#
# Plain 'decentralized', not the yaml's 'decentralized_evaluation': launch_sim.sh
# already opens its own evaluator pane, which evaluation.sh points at this repo's
# evaluator. Leaving the launcher's evaluator on too would run two of them, writing
# reports over each other - the same reason replay_bag_a.sh downgrades the mode.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

exec env DRONE_RUN_MODE="${DRONE_RUN_MODE:-decentralized}" BAG_PROFILE=none \
    "$DEPLOY/launch_triangulation_node.sh" "$@"
