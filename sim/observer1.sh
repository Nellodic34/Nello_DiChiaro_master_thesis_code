#!/usr/bin/env bash
# Sim observer1, running THIS repo's observer_node. The drone name comes from the
# sim's uav_roles, so nothing here needs editing when that world changes.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

source "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ensure_compressed "$SIM_IMG_OBS1"

# publish_camera_static_tf false: the sim already publishes the camera TF chain, and
# this repo would publish its own with the real drone's mount offsets on top of it.
# camera_frame_override is left alone - the YAML already has it empty, which makes the
# node take the frame from the sim's own CameraInfo.
exec env UAV_NAME="$OBSERVER1_NAME" OBSERVER_ROLE=observer1 \
    "$DEPLOY/launch_observer_drone.sh" \
    -r "$DEPLOY_IMG_OBS1":="$SIM_IMG_OBS1/compressed" \
    -r "$DEPLOY_INFO_OBS1":="$SIM_INFO_OBS1" \
    -p publish_camera_static_tf:=false \
    -p device:=cuda \
    "$@"
