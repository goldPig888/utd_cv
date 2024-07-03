import argparse
from _init_paths import *
from lib.RosbagExtractor import RosbagExtractor


def args_parser():
    parser = argparse.ArgumentParser(description="Extract rosbag data")
    parser.add_argument(
        "--rosbag",
        type=str,
        required=True,
        help="Path to the rosbag file",
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=None,
        help="Path to the extrinsics file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    rosbag_file = Path(args.rosbag).resolve()
    extrinsics_file = (
        Path(args.extrinsics).resolve()
        if args.extrinsics
        else sorted(
            [
                f
                for f in Path((PROJ_ROOT / "data/calibration/extrinsics")).glob(
                    "extrinsics_*/extrinsics.json"
                )
            ]
        )[-1]
    )

    # Initialize the RosbagExtractor
    extractor = RosbagExtractor()

    # Extract the rosbag
    extractor.extract_bag(rosbag_file, extrinsics_file)
