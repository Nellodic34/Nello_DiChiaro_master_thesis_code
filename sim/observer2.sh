#!/usr/bin/env bash
# Sim observer2, running THIS repo's observer_node. The drone name comes from the
# sim's uav_roles, so nothing here needs editing when that world changes.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

source "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ensure_compressed "$SIM_IMG_OBS2"

exec env UAV_NAME="$OBSERVER2_NAME" OBSERVER_ROLE=observer2 \
    "$DEPLOY/launch_observer_drone.sh" \
    -r "$DEPLOY_IMG_OBS2":="$SIM_IMG_OBS2/compressed" \
    -r "$DEPLOY_INFO_OBS2":="$SIM_INFO_OBS2" \
    -p publish_camera_static_tf:=false \
    -p device:=cuda \
    "$@"
