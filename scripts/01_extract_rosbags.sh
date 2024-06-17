#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu


CURR_DIR=$(realpath $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
source ${CURR_DIR}/00_config.sh

# Path to the script that extracts the rosbag
SCRIPT_FILE="${PROJ_DIR}/tools/01_extract_rosbag.py"

# Find all rosbags in the data/rosbags directory
ALL_BAGS=$(find ${PROJ_DIR}/data/rosbags -name "*.bag" | sort)


# Extract the rosbag
for BAG in ${ALL_BAGS[@]} ; do
    echo "###############################################################################"
    echo "# Extracting rosbag ${BAG}"
    echo "###############################################################################"
    python ${SCRIPT_FILE} \
        --rosbag ${BAG} \
        --extrinsics ${PROJ_DIR}/data/calibration/extrinsics/extrinsics_20240611/extrinsics.json
done