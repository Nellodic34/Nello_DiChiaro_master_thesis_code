#!/usr/bin/env bash

set -eo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

exec env OBSERVER_ROLE=observer1 "$SCRIPT_DIR/launch_observer_drone.sh" "$@"