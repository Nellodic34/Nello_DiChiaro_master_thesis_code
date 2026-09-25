#!/usr/bin/env bash
# Sim live monitor, from THIS repo. Its launcher opens no --ros-args block of its
# own, so sim time has to be passed here or the plot compares bag stamps to wallclock.
#
# The two observer remaps are what puts the observer triangles back on the bird's eye
# view: live_monitor.py declares no ROS parameters, and builds those subscriptions as
# /{uav_roles observer}/{odom_topic_suffix} from the DEPLOY yaml - so in sim it waits
# on /uav11 and /uav8, which nobody publishes. Target and estimate come from /target/...
# and are name-independent, which is why only the triangles were missing.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/common.sh"

exec "$DEPLOY/launch_live_monitor.sh" --ros-args \
    -p use_sim_time:=true \
    -r "$DEPLOY_ODOM_OBS1":="$SIM_ODOM_OBS1" \
    -r "$DEPLOY_ODOM_OBS2":="$SIM_ODOM_OBS2" \
    "$@"
