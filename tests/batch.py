import pandas as pd
import numpy as np
import torch
from pathlib import Path
from commons import *
from manopth.manolayer import ManoLayer
import matplotlib.pyplot as plt

class dataLoader:
    def __init__(self, sequence_folder) -> None:
        self.data_folder = Path(sequence_folder).resolve()
        self.calib_folder = self.data_folder.parent.parent / "calibration"
        self.load_metadata()
        self.load_intrinsics()
        self.load_extrinsics()
        self.load_mano_model()


    def load_metadata(self):
        meta_file = self.data_folder / "meta.json"
        data = read_data_from_json(meta_file)
        self.imgs = data["realsense"]["serials"]
        self.w = data["realsense"]["width"]
        self.h = data["realsense"]["height"]
        self.extr_file = self.calib_folder / "extrinsics" / data["calibration"]["extrinsics"] / "extrinsics.json"
        self.subject_id = data["calibration"]["mano"]
        self.mano_file = self.calib_folder / "mano" / self.subject_id / "mano.json"
        self.object_ids = data["object_ids"]
        self.mano_sides = data["mano_sides"]
        self.num_frames = data["num_frames"]


    def load_intrinsics(self):
        self.intrinsics = {s: read_K_matrix_from_json(
            self.calib_folder / "intrinsics" / f"{s}_{self.w}x{self.h}.json"
        ) for s in self.imgs}


    def load_extrinsics(self):
        extrinsics, rs_master = read_extrinsics_from_json(self.extr_file)
        self.rs_master = rs_master
        tag1_inv = np.linalg.inv(extrinsics["tag_1"])
        self.extrinsics2master = {s: extrinsics[s] for s in self.imgs}
        self.extrinsics2world = {s: tag1_inv @ extrinsics[s] for s in self.imgs}


    def load_mano_model(self):
        self.mano_layer = ManoLayer(
            center_idx=0,
            flat_hand_mean=True,
            ncomps=45,
            side="left",
            mano_root=PROJ_ROOT / "config/mano_models",
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
            robust_rot=True,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mano_layer = self.mano_layer.to(self.device)


    def get_mano_keypoints(self, hand_pose, hand_shape):
        hand_verts, hand_joints = self.mano_layer(torch.from_numpy(hand_pose), torch.from_numpy(hand_shape))
        return hand_verts.detach().cpu().numpy(), hand_joints.detach().cpu().numpy()


    def get_image(self, id, img, img_type="color"):
        suffix = "jpg" if img_type == "color" else "png"
        img_file = self.data_folder / img / f"{img_type}_{id:06d}.{suffix}"
        if not img_file.exists():
            return np.zeros((self.h, self.w, 3 if img_type == "color" else 1), dtype=np.uint8)
        return read_rgb_image(img_file) if img_type == "color" else read_depth_image(img_file)


    def get_points(self, id, img, space="camera"):
        depth = self.get_image(id, img, "depth").astype(np.float32) / 1000.0
        K = self.intrinsics[img]
        RT = self.extrinsics2master[img] if space == "camera" else self.extrinsics2world[img]
        return deproject_depth_image(depth, K, RT)


    def visualize_point_cloud(self, id, space="camera"):
        point_clouds = []
        colors = []
        for img in self.imgs:
            points = self.get_points(id, img, space)
            color = self.get_image(id, img, "color").reshape(-1, 3).astype(np.float32) / 255.0
            point_clouds.append(points)
            colors.append(color)

        point_clouds = np.vstack(point_clouds)
        colors = np.vstack(colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw([pcd])



class HandPoseOptimizer:
    def __init__(self, sequence_loader: dataLoader):
        self.sequence_loader = sequence_loader
        self.device = self.sequence_loader.device
        self.mano_layer = self.sequence_loader.mano_layer


    def mano_layer_forward(self, mano_pose, mano_shape):
        verts, joints = self.mano_layer(mano_pose[:, :48], mano_shape, mano_pose[:, 48:])
        verts /= 1000.0
        joints /= 1000.0
        return verts, joints


    def loss_3d_keypoints(self, mano_joints, target_kpts):
        mse_loss = torch.nn.MSELoss(reduction="sum")
        loss = mse_loss(mano_joints, target_kpts)
        loss /= mano_joints.shape[0]
        return loss


    def optimize_hand_pose(self, target_kpts):
        optim_pose = torch.zeros((1, 51), dtype=torch.float32, device=self.device)
        optim_pose[:, 48:] = target_kpts[:, 0, :].clone()
        optim_pose.requires_grad = True

        data = read_data_from_json(self.sequence_loader.mano_file)
        shape_params = torch.tensor(data["betas"], dtype=torch.float32).to(self.device).unsqueeze(0)

        optimizer = torch.optim.Adam([optim_pose], lr=0.001)

        loss_history = []
        total_steps = 10000

        for step in range(total_steps):
            optimizer.zero_grad()
            mano_verts, mano_joints = self.mano_layer_forward(optim_pose, shape_params)

            loss = self.loss_3d_keypoints(mano_joints, target_kpts)
            loss.backward()

            optimizer.step()

            if (step+1) % 500 == 0:
                print(f"Step {step+1:06d}/{total_steps:06d}, Loss {loss.item():11.8f}")

            loss_history.append(loss.item())

        plt.plot(loss_history)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Optimization Loss")
        plt.show()

        return mano_verts, mano_joints


    def visualize_hand_pose(self, mano_verts, mano_joints, target_kpts):
        mano_faces = self.mano_layer.th_faces.detach().cpu().numpy()

        mano_verts = mano_verts[0].detach().cpu().numpy()
        mano_joints = mano_joints[0].detach().cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mano_verts)
        mesh.triangles = o3d.utility.Vector3iVector(mano_faces)
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        mesh.compute_vertex_normals()
        mesh.normalize_normals()

        # joints = red spheres
        joint_mesh = o3d.geometry.TriangleMesh()
        for joint in mano_joints:
            joint_mesh += o3d.geometry.TriangleMesh.create_sphere(radius=0.003).translate(joint)
        joint_mesh.paint_uniform_color([1.0, 0.0, 0.0])

        # keypoints = blue spheres
        kpts_mesh = o3d.geometry.TriangleMesh()
        for kpt in target_kpts[0].cpu().numpy():
            kpts_mesh += o3d.geometry.TriangleMesh.create_sphere(radius=0.003).translate(kpt)
        kpts_mesh.paint_uniform_color([0.0, 0.0, 1.0])

        o3d.visualization.draw([mesh, joint_mesh, kpts_mesh])



if __name__ == "__main__":
    sequence_folder = PROJ_ROOT / "data/recordings/20231022_193630"
    loader = dataLoader(sequence_folder)

    kpts_3d = np.load(PROJ_ROOT / "data/recordings/20231022_193630/hand_keypoints_3d.npy")
    target_kpts = torch.tensor(kpts_3d[300, 1], dtype=torch.float32).unsqueeze(0).to(loader.device)

    optimizer = HandPoseOptimizer(loader)
    mano_verts, mano_joints = optimizer.optimize_hand_pose(target_kpts)

    optimizer.visualize_hand_pose(mano_verts, mano_joints, target_kpts)
