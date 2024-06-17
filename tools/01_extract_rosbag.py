from _init_paths import *

from lib.RosbagExtractor import RosbagExtractor

if __name__ == "__main__":
    extrinsics_file = (
        PROJ_ROOT / "data/calibration/extrinsics/extrinsics_20240611/extrinsics.json"
    )
    rosbag_file = PROJ_ROOT / "data/rosbags/may/20240612/20240612_095552.bag"

    # Initialize the RosbagExtractor
    extractor = RosbagExtractor()

    # Extract the rosbag
    extractor.extract_bag(rosbag_file, extrinsics_file)
