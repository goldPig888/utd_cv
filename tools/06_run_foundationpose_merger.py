import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import trimesh
from scipy.interpolate import interp1d, CubicSpline
import argparse

from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader

# Set spawn method for multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def get_best_fd_pose(
    pose_folder,
    frame_id,
    serials,
    cam_RTs,
    object_id,
    iterations=500,
    lr=0.001,
    rot_threshold=10,
    tran_threshold=0.01,
):
    def normalize_quaternion(q):
        norm = torch.norm(q, dim=-1, keepdim=True)
        return q / norm

    def quaternion_conjugate(q):
        return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

    def quaternion_multiply(q1, q2):
        x1, y1, z1, w1 = q1.unbind(dim=-1)
        x2, y2, z2, w2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w], dim=-1)

    def quaternion_difference(q1, q2):
        q1 = normalize_quaternion(q1)
        q2 = normalize_quaternion(q2)

        q1_conjugate = quaternion_conjugate(q1)
        q_diff = quaternion_multiply(q1_conjugate, q2)

        q_diff_w = torch.clamp(
            q_diff[..., 3], -1.0 + 1e-6, 1.0 - 1e-6
        )  # Slightly stricter clamping to avoid acos issues
        angle = 2 * torch.acos(q_diff_w)

        angle = torch.where(angle > torch.pi, 2 * torch.pi - angle, angle)

        return angle

    def loss_quat(q1, q2):
        """
        Compute the quaternion distance between one quaternion q1 and a set of quaternions q2.
        q1: shape (4,)
        q2: shape (N, 4)
        """
        q1 = q1.unsqueeze(0)  # Make q1 shape (1, 4)
        angle = quaternion_difference(q1, q2)
        return angle

    def loss_tran(t1, t2):
        return torch.norm(t1 - t2, p=2, dim=1)

    def get_total_losses(rot, tran, target_rots, target_trans):
        rotation_losses = loss_quat(rot, target_rots)
        translation_losses = loss_tran(tran, target_trans)
        return rotation_losses, translation_losses

    def get_optimal_pose(rot, tran, all_rots, all_trans):
        opt_rot = torch.from_numpy(rot).to(device).requires_grad_(True)
        opt_tran = torch.from_numpy(tran).to(device).requires_grad_(True)

        all_rots = torch.from_numpy(all_rots).to(device)
        all_trans = torch.from_numpy(all_trans).to(device)

        rot_optimizer = torch.optim.Adam([opt_rot], lr=lr)
        tran_optimizer = torch.optim.Adam([opt_tran], lr=lr)

        for i in range(iterations):
            rot_optimizer.zero_grad()
            tran_optimizer.zero_grad()

            rot_losses, tran_losses = get_total_losses(
                opt_rot, opt_tran, all_rots, all_trans
            )
            mean_rot_loss = torch.mean(rot_losses)
            mean_tran_loss = torch.mean(tran_losses)

            mean_rot_loss.backward()
            mean_tran_loss.backward()

            rot_optimizer.step()
            tran_optimizer.step()

            # Normalize the quaternion part to ensure it remains valid
            opt_rot.data = normalize_quaternion(opt_rot.data)

        rot_losses, tran_losses = get_total_losses(
            opt_rot, opt_tran, all_rots, all_trans
        )

        rot_losses = rot_losses.detach().cpu().numpy()
        tran_losses = tran_losses.detach().cpu().numpy()
        opt_rot = opt_rot.detach().cpu().numpy()
        opt_tran = opt_tran.detach().cpu().numpy()

        return opt_rot, opt_tran, rot_losses, tran_losses

    # convert rot_thresh to radians
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd_poses = []
    best_fd_pose = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)

    for serial in serials:
        pose_file = pose_folder / f"ob_in_cam/{object_id}/{serial}/{frame_id:06d}.txt"
        if not pose_file.exists():
            continue

        ob_in_cam = np.loadtxt(pose_file, dtype=np.float32)

        if np.all(ob_in_cam == -1):
            continue

        cam_idx = serials.index(serial)
        ob_in_world = cam_RTs[cam_idx] @ ob_in_cam
        ob_in_world = mat_to_quat(ob_in_world)
        fd_poses.append(ob_in_world)

    if len(fd_poses) < 4:
        return best_fd_pose

    fd_poses = np.stack(fd_poses, axis=0, dtype=np.float32)

    # Generate the candidate information
    opt_rots, opt_trans, rot_losses, tran_losses = [], [], [], []
    for fd_pose in fd_poses:
        opt_rot, opt_tran, rot_loss, tran_loss = get_optimal_pose(
            fd_pose[:4], fd_pose[4:], fd_poses[:, :4], fd_poses[:, 4:]
        )
        opt_rots.append(opt_rot)
        opt_trans.append(opt_tran)
        rot_losses.append(rot_loss)
        tran_losses.append(tran_loss)

        # tqdm.write(
        #     f"rot_loss: {[f'{l:.6f}' for l in rot_loss]}, tran_loss: {[f'{l:.6f}' for l in tran_loss]}"
        # )

    # Find the best pose by RANSAC
    best_rot = np.array([-1, -1, -1, -1], dtype=np.float32)
    best_tran = np.array([-1, -1, -1], dtype=np.float32)
    best_rot_inliers = 0
    best_tran_inliers = 0
    best_rot_losses = np.inf
    best_tran_losses = np.inf

    for opt_rot, opt_tran, rot_loss, tran_loss in zip(
        opt_rots, opt_trans, rot_losses, tran_losses
    ):
        rot_inliers = np.sum(rot_loss < rot_threshold)
        tran_inliers = np.sum(tran_loss < tran_threshold)

        if rot_inliers > best_rot_inliers:
            best_rot_inliers = rot_inliers
            best_rot = opt_rot
            best_rot_losses = rot_loss

        if tran_inliers > best_tran_inliers:
            best_tran_inliers = tran_inliers
            best_tran = opt_tran
            best_tran_losses = tran_loss

        if rot_inliers == best_rot_inliers and np.mean(rot_loss) < np.mean(
            best_rot_losses
        ):
            best_rot = opt_rot
            best_rot_losses = rot_loss

        if tran_inliers == best_tran_inliers and np.mean(tran_loss) < np.mean(
            best_tran_losses
        ):
            best_tran = opt_tran
            best_tran_losses = tran_loss

    if best_rot_inliers < 2 or best_tran_inliers < 2:
        return best_fd_pose

    valid_rots = fd_poses[:, :4][best_rot_losses < rot_threshold]
    valid_trans = fd_poses[:, 4:][best_tran_losses < tran_threshold]

    best_rot, best_tran, best_rot_losses, best_tran_losses = get_optimal_pose(
        best_rot, best_tran, valid_rots, valid_trans
    )

    best_fd_pose = np.concatenate([best_rot, best_tran], axis=0)

    return best_fd_pose


def slerp(q0, q1, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.
    """
    dot = torch.sum(q0 * q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.95:
        result = q0 + t * (q1 - q0)
        return result / torch.norm(result)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


def complete_poses_interp1d(poses):
    if isinstance(poses, list):
        poses = np.stack(poses, axis=0, dtype=np.float32)

    N, D = poses.shape
    assert D == 7, "Each pose should have 7 values: (qx, qy, qz, qw, tx, ty, tz)"

    valid_indices = np.where(~np.all(poses == -1, axis=1))[0]
    invalid_indices = np.where(np.all(poses == -1, axis=1))[0]

    if len(invalid_indices) == 0:
        return poses

    # Interpolate translation components
    for i in range(4, 7):
        valid_values = poses[valid_indices, i]
        f = interp1d(
            valid_indices, valid_values, kind="linear", fill_value="extrapolate"
        )
        poses[invalid_indices, i] = f(invalid_indices)

    # Interpolate quaternions
    for invalid_idx in invalid_indices:
        before_indices = valid_indices[valid_indices < invalid_idx]
        after_indices = valid_indices[valid_indices > invalid_idx]

        if before_indices.size > 0 and after_indices.size > 0:
            before_idx = before_indices[-1]
            after_idx = after_indices[0]

            t = (invalid_idx - before_idx) / (after_idx - before_idx)
            q1 = poses[before_idx, :4]
            q2 = poses[after_idx, :4]
            q_interp = slerp(torch.tensor(q1), torch.tensor(q2), t).numpy()
            poses[invalid_idx, :4] = q_interp
        elif before_indices.size > 0:
            before_idx = before_indices[-1]
            poses[invalid_idx, :4] = poses[before_idx, :4]
        elif after_indices.size > 0:
            after_idx = after_indices[0]
            poses[invalid_idx, :4] = poses[after_idx, :4]
        else:
            raise ValueError("No valid poses to interpolate from.")

    return poses


def complete_poses_cubic(poses):
    if isinstance(poses, list):
        poses = np.stack(poses, axis=0, dtype=np.float32)

    N, D = poses.shape
    assert D == 7, "Each pose should have 7 values: (qx, qy, qz, qw, tx, ty, tz)"

    valid_indices = np.where(~np.all(poses == -1, axis=1))[0]
    invalid_indices = np.where(np.all(poses == -1, axis=1))[0]

    if len(invalid_indices) == 0:
        return poses

    # Interpolate translation components
    for i in range(4, 7):
        valid_values = poses[valid_indices, i]
        f = CubicSpline(valid_indices, valid_values)
        poses[invalid_indices, i] = f(invalid_indices)

    # Interpolate quaternions
    for invalid_idx in invalid_indices:
        before_indices = valid_indices[valid_indices < invalid_idx]
        after_indices = valid_indices[valid_indices > invalid_idx]

        if before_indices.size > 0 and after_indices.size > 0:
            before_idx = before_indices[-1]
            after_idx = after_indices[0]

            t = (invalid_idx - before_idx) / (after_idx - before_idx)
            q1 = poses[before_idx, :4]
            q2 = poses[after_idx, :4]
            q_interp = slerp(torch.tensor(q1), torch.tensor(q2), t).numpy()
            poses[invalid_idx, :4] = q_interp
        elif before_indices.size > 0:
            before_idx = before_indices[-1]
            poses[invalid_idx, :4] = poses[before_idx, :4]
        elif after_indices.size > 0:
            after_idx = after_indices[0]
            poses[invalid_idx, :4] = poses[after_idx, :4]
        else:
            raise ValueError("No valid poses to interpolate from.")

    return poses


def get_rendered_image(rgb_images, object_mesh, object_pose, cam_Ks, cam_RTs, serials):
    height, width = rgb_images[0].shape[:2]
    r = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    # Add root node
    root_node = scene.add_node(pyrender.Node())

    # Add camera nodes
    cam_nodes = [
        scene.add(
            pyrender.IntrinsicsCamera(
                fx=cam_K[0, 0],
                fy=cam_K[1, 1],
                cx=cam_K[0, 2],
                cy=cam_K[1, 2],
                znear=0.01,
                zfar=10.0,
            ),
            pose=cam_RTs[i] @ cvcam_in_glcam,
            name=f"camera_{i}",
            parent_node=root_node,
        )
        for i, cam_K in enumerate(cam_Ks)
    ]

    # Add object node
    object_node = scene.add(
        pyrender.Mesh.from_trimesh(object_mesh, smooth=False),
        name="object",
        pose=np.eye(4),
        parent_node=root_node,
    )
    scene.set_pose(object_node, object_pose)

    vis = []
    for idx, cam_node in enumerate(cam_nodes):
        scene.main_camera_node = cam_node
        r_color, _ = r.render(scene, flags=pyrender.RenderFlags.ALL_SOLID)
        vis.append(cv2.addWeighted(rgb_images[idx], 0.25, r_color, 0.75, 0))
    r.delete()

    vis = display_images(vis, serials, return_array=True)

    return vis


def main():
    start_time = time.time()

    fd_pose_folder = sequence_folder / "processed/foundationpose"

    logger.info("  - Merging foundation poses by RANSAC...")
    save_fd_poses_merged_file = fd_pose_folder / "fd_poses_merged_raw.npy"
    if save_fd_poses_merged_file.exists():
        logger.info("    ** Merged foundation poses already exist, loading...")
        best_fd_poses = np.load(save_fd_poses_merged_file)
    else:
        best_fd_poses = [None] * num_frames
        tqbar = tqdm(total=num_frames, ncols=80, colour="green")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    get_best_fd_pose,
                    fd_pose_folder,
                    frame_id,
                    serials,
                    cam_RTs,
                    object_id,
                    lr=0.001,
                    iterations=150,
                    rot_threshold=0.03,  # radians
                    tran_threshold=0.01,  # meters
                ): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                best_fd_poses[futures[future]] = future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

        best_fd_poses = np.stack(best_fd_poses, axis=0, dtype=np.float32)
        logger.info(f"    ** best_fd_poses: {best_fd_poses.shape}")

        # Save the merged foundation poses
        np.save(save_fd_poses_merged_file, best_fd_poses)

    # Complete the missing poses
    save_fd_poses_completed_file = fd_pose_folder / "fd_poses_interpolated.npy"
    best_fd_poses = complete_poses_interp1d(best_fd_poses)
    # best_fd_poses = complete_poses_cubic(best_fd_poses)
    np.save(save_fd_poses_completed_file, best_fd_poses)

    # Draw rendered images with the merged foundation poses
    logger.info("  - Generating vis images...")
    tqbar = tqdm(total=num_frames, ncols=80, colour="green")
    vis_images = [None] * num_frames
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                get_rendered_image,
                loader.get_rgb_image(frame_id),
                object_mesh,
                quat_to_mat(best_fd_poses[frame_id]),
                cam_Ks,
                cam_RTs,
                serials,
            ): frame_id
            for frame_id in range(num_frames)
        }
        for future in concurrent.futures.as_completed(futures):
            vis_images[futures[future]] = future.result()
            tqbar.update()
            tqbar.refresh()
        del futures
    tqbar.close()

    logger.info("  - Saving vis images...")
    tqbar = tqdm(total=num_frames, ncols=80, colour="green")
    save_vis_folder = fd_pose_folder / "vis" / "fd_poses_merged"
    save_vis_folder.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                write_rgb_image,
                save_vis_folder / f"vis_{frame_id:06d}.png",
                vis_images[frame_id],
            ): frame_id
            for frame_id in range(num_frames)
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            tqbar.update()
            tqbar.refresh()
        del futures
    tqbar.close()

    logger.info("  - Saving vis video...")
    create_video_from_rgb_images(
        fd_pose_folder / "fd_poses_merged.mp4", vis_images, fps=30
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f">>>>>>>>>> Done!!! ({elapsed_time:.2f} seconds)<<<<<<<<<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, required=True, help="Path to the sequence folder"
    )
    args = parser.parse_args()
    sequence_folder = Path(args.sequence_folder).resolve()

    logger = get_logger("FoundationPoseMerger")

    loader = SequenceLoader(sequence_folder)
    width = loader.width
    height = loader.height
    serials = loader.serials
    num_frames = loader.num_frames
    cam_Ks = loader.Ks.cpu().numpy()
    cam_RTs = loader.extrinsics2world.cpu().numpy()
    object_id = loader.object_id
    object_mesh = trimesh.load_mesh(loader.object_textured_mesh_file, process=False)

    main()
