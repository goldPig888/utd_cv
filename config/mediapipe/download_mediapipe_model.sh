#!/bin/bash

CURR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Download the Hand Landmark model and the palm detection model.
echo "Downloading the Hand Landmark model and the palm detection model..."
rm -rf ${CURR_DIR}/hand_landmarker.task && \
wget -q --show-progress https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O ${CURR_DIR}/hand_landmarker.task
