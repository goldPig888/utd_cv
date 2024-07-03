#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu


CURR_DIR=$(realpath $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))
PROJ_DIR=$(dirname ${CURR_DIR})
SCRIPT_FILE="${PROJ_DIR}/tools/04_run_mask_label_toolkit.py"

python ${SCRIPT_FILE}
