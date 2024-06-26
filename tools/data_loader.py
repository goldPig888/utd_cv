import open3d as o3d
import open3d.core as o3c
import torch
from torch.utils.dlpack import to_dlpack
import argparse

from _init_paths import *
from lib.SequenceLoader import SequenceLoader


def args_parser():
    parser = argparse.ArgumentParser(description="Load sequence data")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        required=True,
        help="Path to the sequence folder",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    sequence_folder = Path(args.sequence_folder).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = SequenceLoader(sequence_folder, device=device)

    frame_id = 100
    loader.step_by_frame_id(frame_id)
    masks = loader.pcd_masks
    points = loader.pcd_points[masks]
    colors = loader.pcd_colors[masks]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor.from_dlpack(to_dlpack(points.cpu()))
    pcd.point.colors = o3c.Tensor.from_dlpack(to_dlpack(colors.cpu()))

    o3d.visualization.draw([pcd])
