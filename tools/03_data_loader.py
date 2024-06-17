import open3d as o3d
import open3d.core as o3c

from torch.utils.dlpack import to_dlpack

from _init_paths import *
from lib.Utils import display_images
from lib.SequenceLoader import SequenceLoader


if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/recordings/may_20240612_095552"
    loader = SequenceLoader(sequence_folder)

    frame_id = 100
    loader.step_by_frame_id(frame_id)
    masks = loader.pcd_masks
    points = loader.pcd_points[masks]
    colors = loader.pcd_colors[masks]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor.from_dlpack(to_dlpack(points.cpu()))
    pcd.point.colors = o3c.Tensor.from_dlpack(to_dlpack(colors.cpu()))

    o3d.visualization.draw([pcd])
