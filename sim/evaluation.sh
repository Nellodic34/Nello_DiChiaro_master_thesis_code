#!/usr/bin/env bash
# Sim prediction evaluator, from THIS repo.
#
# launch_sim.sh opens an evaluator pane whenever the SIM yaml has
# evaluation.auto_start_with_launch_sim: true - and it does. Without this shim that
# pane runs the OTHER tree's evaluator, so the run would be scored by a node this
# repo does not ship. That is also why triangulation.sh uses plain 'decentralized':
# 'decentralized_evaluation' would start a second evaluator inside the launcher, and
# two of them write reports over each other.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

exec "$DEPLOY/run_prediction_evaluation.sh" "$@"
