# Shared by the sim shims. Sourced, not executed.
#
# Maps this repo's deploy identity (uav11/uav8/uav2, RealSense topics) onto the
# FlightForge sim's (uav1/uav2/uav3, /rgb/image_raw). Roles and UAV names are env
# knobs the launchers already honour; topic names are NOT - several nodes read them
# straight from their yaml and declare no parameter for them, so they are redirected
# with ROS remaps instead. That leaves the tracked YAML untouched, which is the whole
# point of this directory.
#
# Nothing here is hardcoded: BOTH sides are read from their own yaml. Writing either
# side by hand is how the live monitor's observer markers went missing - the deploy
# names had to match uav_roles exactly, and a literal cannot.

DEPLOY="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

export VENV_ACTIVATE="${VENV_ACTIVATE_SIM:-$DEPLOY/.venv/bin/activate}"

# The launchers source the venv only "if [ -f ]", so a missing one is not an error
# there: the nodes just start without ultralytics and die on an import nobody
# connects back to this. A worktree hits exactly that - .venv is gitignored, so it
# exists only in the main checkout. Fail here instead, while the cause is still visible.
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "sim: venv assente: $VENV_ACTIVATE" >&2
    echo "sim: esegui l'harness dal checkout principale, o passa VENV_ACTIVATE_SIM=<path>/activate" >&2
    exit 1
fi

export USE_SIM_TIME=true

SIM_TREE="${SIM_TREE:-$HOME/git/training/mrs_uav_flightforge_simulator/tmux/one_drone}"
SIM_CONFIG="$SIM_TREE/config/runtime_stack.yaml"
DEPLOY_CONFIG="$DEPLOY/config/runtime_stack.yaml"

# Same python+yaml idiom the launchers already use before the venv exists.
cfg_get() {
    python3 -c "
import yaml
try:
    val = yaml.safe_load(open('$1'))
    for key in '$2'.split('.'):
        val = val[key]
    print(val if val is not None else '')
except Exception:
    print('')
"
}

need() {
    if [ -z "$2" ]; then
        echo "sim: '$1' non leggibile dai runtime_stack.yaml" >&2
        exit 1
    fi
}

# ── Sim side: the drones that world actually spawns ──────────────────────────
export OBSERVER1_NAME="${OBSERVER1_NAME:-$(cfg_get "$SIM_CONFIG" uav_roles.observer1)}"
export OBSERVER2_NAME="${OBSERVER2_NAME:-$(cfg_get "$SIM_CONFIG" uav_roles.observer2)}"
export TARGET_NAME="${TARGET_NAME:-$(cfg_get "$SIM_CONFIG" uav_roles.target)}"
need uav_roles.observer1 "$OBSERVER1_NAME"
need uav_roles.observer2 "$OBSERVER2_NAME"
need uav_roles.target "$TARGET_NAME"

# The shape observer_node falls back to when the yaml override is empty, which is how
# the sim tree runs its cameras.
SIM_IMG_OBS1="/$OBSERVER1_NAME/rgb/image_raw"
SIM_IMG_OBS2="/$OBSERVER2_NAME/rgb/image_raw"
SIM_INFO_OBS1="/$OBSERVER1_NAME/rgb/camera_info"
SIM_INFO_OBS2="/$OBSERVER2_NAME/rgb/camera_info"

# ── Deploy side: the names the nodes will actually subscribe to ──────────────
DEPLOY_OBS1="$(cfg_get "$DEPLOY_CONFIG" uav_roles.observer1)"
DEPLOY_OBS2="$(cfg_get "$DEPLOY_CONFIG" uav_roles.observer2)"
DEPLOY_IMG_OBS1="$(cfg_get "$DEPLOY_CONFIG" vision.main_image_topic)"
DEPLOY_IMG_OBS2="$(cfg_get "$DEPLOY_CONFIG" vision.observer_image_topic)"
ODOM_SUFFIX="$(cfg_get "$DEPLOY_CONFIG" evaluation.odom_topic_suffix)"
need uav_roles.observer1 "$DEPLOY_OBS1"
need uav_roles.observer2 "$DEPLOY_OBS2"
need vision.main_image_topic "$DEPLOY_IMG_OBS1"
need vision.observer_image_topic "$DEPLOY_IMG_OBS2"
need evaluation.odom_topic_suffix "$ODOM_SUFFIX"

# CameraInfo: the explicit override wins, otherwise derive it exactly the way
# observer_node does - drop '/compressed', drop the trailing '/image_raw', append.
derive_info() {
    local base="${1%/compressed}"
    base="${base%/image_raw}"
    echo "$base/camera_info"
}
DEPLOY_INFO_OBS1="$(cfg_get "$DEPLOY_CONFIG" vision.main_camera_info_topic)"
DEPLOY_INFO_OBS2="$(cfg_get "$DEPLOY_CONFIG" vision.observer_camera_info_topic)"
[ -n "$DEPLOY_INFO_OBS1" ] || DEPLOY_INFO_OBS1="$(derive_info "$DEPLOY_IMG_OBS1")"
[ -n "$DEPLOY_INFO_OBS2" ] || DEPLOY_INFO_OBS2="$(derive_info "$DEPLOY_IMG_OBS2")"

# live_monitor.py declares no ROS parameters at all, so its observer markers can only
# be redirected by remap. It builds them as /{uav_roles observer}/{odom_topic_suffix}.
DEPLOY_ODOM_OBS1="/$DEPLOY_OBS1/$ODOM_SUFFIX"
DEPLOY_ODOM_OBS2="/$DEPLOY_OBS2/$ODOM_SUFFIX"
SIM_ODOM_OBS1="/$OBSERVER1_NAME/$ODOM_SUFFIX"
SIM_ODOM_OBS2="/$OBSERVER2_NAME/$ODOM_SUFFIX"

# The deploy weights are trained on real footage; the sim needs the sim-trained ones.
export MODEL_PATH="${MODEL_PATH:-$SIM_TREE/dataset_tools/weights_sim/best_openvino_model}"

# observer_node picks CompressedImage vs Image from the '/compressed' suffix of the
# CONFIGURED name, not of the remapped one, so it still expects compressed after the
# remap. FlightForge may or may not publish a compressed transport; start a
# republisher only when it does not.
ensure_compressed() {
    local raw="$1" comp="$1/compressed"
    if ros2 topic list 2>/dev/null | grep -qx "$comp"; then
        return 0
    fi
    echo "sim: $comp assente, avvio republish da $raw"
    ros2 run image_transport republish raw compressed \
        --ros-args -r in:="$raw" -r out/compressed:="$comp" &
}
