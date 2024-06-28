#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu


CURR_DIR=$(realpath $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
source ${CURR_DIR}/00_config.sh

# Path to the script that extracts the rosbag
SCRIPT_FILE="${PROJ_DIR}/tools/02-2_run_hand_3d_joints_generation.py"

# Find all rosbags in the data/rosbags directory
ALL_SEQUENCES=(
data/recordings/ida_20240617_101133
data/recordings/isaac_20240617_102035
data/recordings/lyndon_20240617_102549
data/recordings/may_20240617_101936
data/recordings/nicole_20240617_102128
data/recordings/reanna_20240617_102436
data/recordings/rebecca_20240617_100917
)


# Extract the rosbag
for SEQUENCE in ${ALL_SEQUENCES[@]} ; do
    echo "###############################################################################"
    echo "# Running MP Handmarks Detection on ${SEQUENCE}"
    echo "###############################################################################"
    python ${SCRIPT_FILE} \
        --sequence_folder ${PROJ_DIR}/${SEQUENCE}
done