import argparse
import itertools
from scipy.interpolate import CubicSpline, interp1d

from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader


# Set the start method to 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def runner_hand_3d_joints_ransac_torch(
    mp_handmarks, Ms, threshold=10.0, iterations=100, lr=1e-3
):
    def triangulate_handmarks_batch(p1, p2, M1, M2):
        """
        Triangulate handmarks from two camera views using PyTorch for batch processing.

        Parameters:
        - p1: Batch of points from the first camera, torch tensor of shape (B, 21, 2).
        - p2: Batch of points from the second camera, torch tensor of shape (B, 21, 2).
        - M1, M2: Corresponding camera projection matrices, torch tensors of shape (B, 3, 4).

        Returns:
        - A torch tensor of shape (B, 21, 3) containing the triangulated 3D handmarks.
        """
        B, N, _ = p1.shape  # B: Batch size, N: Number of handmarks (21)
        X = torch.zeros((B, N, 3), dtype=torch.float32, device=p1.device)

        for i in range(N):  # Process each handmark
            A = torch.zeros((B, 4, 4), dtype=torch.float32, device=p1.device)
            A[:, 0, :] = p1[:, i, 0].unsqueeze(1) * M1[:, 2, :] - M1[:, 0, :]
            A[:, 1, :] = p1[:, i, 1].unsqueeze(1) * M1[:, 2, :] - M1[:, 1, :]
            A[:, 2, :] = p2[:, i, 0].unsqueeze(1) * M2[:, 2, :] - M2[:, 0, :]
            A[:, 3, :] = p2[:, i, 1].unsqueeze(1) * M2[:, 2, :] - M2[:, 1, :]

            # Perform SVD on A
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            # Extract the solution from Vh, normalize by the last element
            X[:, i, :] = Vh[:, -1, :3] / Vh[:, -1, 3].unsqueeze(1)

        return X

    def project_3d_to_2d_parallel(p_3d, Ms):
        """
        Project a batch of 3D points to 2D across multiple camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - projected_2d: Projected 2D points, torch tensor of shape (C, N, 2).
        """
        C = Ms.shape[0]
        N = p_3d.shape[0]
        ones = torch.ones((N, 1), device=p_3d.device, dtype=p_3d.dtype)
        p_3d_hom = (
            torch.cat((p_3d, ones), dim=1).unsqueeze(0).repeat(C, 1, 1)
        )  # Shape: (C, N, 4)
        p_2d_hom = torch.einsum("cij,ckj->cki", Ms, p_3d_hom)  # Shape: (C, N, 3)
        p_2d = p_2d_hom[:, :, :2] / p_2d_hom[:, :, [2]]  # Normalize
        return p_2d

    def parallel_reprojection_loss(p_3d, uv_coords, Ms):
        """
        Compute the reprojection loss for all 21 joints across all camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - Total reprojection loss, a single scalar value.
        """
        projected_2d = project_3d_to_2d_parallel(p_3d, Ms)
        loss = torch.norm(
            projected_2d - uv_coords, dim=-1
        )  # Compute L2 loss for each camera

        loss = loss.sum()
        loss /= uv_coords.size(0) * p_3d.size(0)  # Normalize by number of cameras
        return loss

    def optimize_all_joints(pts_3d, uv_coords, Ms, lr=1e-3, steps=100):
        """
        Optimize the 3D coordinates of all 21 MANO joints in parallel.

        Parameters:
        - pts_3d: Batch of initial 3D coordinates for all joints, torch tensor of shape (B, N, 3).
        - uv_coords: Observed 2D points for all cameras, torch tensor of shape (C, N, 2).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).
        - lr: Learning rate for the optimizer.
        - steps: Number of optimization steps.

        Returns:
        - Optimized 3D coordinates for all 21 joints, torch tensor of shape (N, 3).
        """
        B, N, _ = pts_3d.shape
        X = torch.zeros(
            (B, 3),
            device=pts_3d.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([X], lr=lr)
        opt_pts = []

        for i in range(N):
            X.data = pts_3d[:, i].clone()
            _uv_coords = uv_coords[:, i].unsqueeze(1).repeat(1, B, 1)
            for _ in range(steps):
                optimizer.zero_grad()
                loss = parallel_reprojection_loss(X, _uv_coords, Ms)
                loss.backward()
                optimizer.step()
            opt_pts.append(X.detach().clone())

        opt_pts = torch.stack(opt_pts)
        return opt_pts

    Ms_ts = torch.from_numpy(Ms).to(dtype=torch.float32, device="cuda")
    handmarks = torch.from_numpy(mp_handmarks).to(dtype=torch.float32, device="cuda")
    hands, num_cams, joints, _ = handmarks.shape

    hand_joints_3d = torch.full((hands, joints, 3), -1, dtype=torch.float32)
    for hand in range(hands):
        marks = handmarks[hand]  # shape (num_cameras, num_joints, 2)
        if torch.all(marks == -1):  # no hand detected
            continue
        valid_views = torch.where(torch.all(marks[..., 0] != -1, dim=1))
        num_valid_views = len(valid_views[0])
        if num_valid_views < 5:  # less than 2 views detected
            continue

        # create candidate 3D points by triangulating 2D points from valid views
        combinations = torch.tensor(
            list(itertools.combinations(valid_views[0].cpu().numpy(), 2)),
            device=marks.device,
        )
        b_pts_3d = triangulate_handmarks_batch(
            marks[combinations[:, 0]],
            marks[combinations[:, 1]],
            Ms_ts[combinations[:, 0]],
            Ms_ts[combinations[:, 1]],
        )

        # optimize each 3D point to minimize reprojection error
        pts_3d_optim = optimize_all_joints(
            b_pts_3d,
            marks[valid_views],
            Ms_ts[valid_views],
            lr=lr,
            steps=iterations,
        )

        best_points = torch.full((joints, 3), -1, dtype=torch.float32)
        for joint in range(joints):
            best_loss = 0.0
            best_point = None
            best_inliers = 0
            for i in range(len(pts_3d_optim[joint])):
                projected_2d = project_3d_to_2d_parallel(
                    pts_3d_optim[joint, i].unsqueeze(0),
                    Ms_ts[valid_views],
                )
                loss = torch.norm(
                    projected_2d - marks[valid_views][:, joint].unsqueeze(1), dim=-1
                )
                valid_mask = torch.where(loss < threshold)[0]
                inliers = len(valid_mask)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint, i]
                    best_loss = loss[valid_mask].mean().item()
                if (
                    best_inliers > 0
                    and inliers == best_inliers
                    and loss[valid_mask].mean().item() < best_loss
                ):
                    best_inliers = inliers
                    best_point = pts_3d_optim[joint, i]
                    best_loss = loss[valid_mask].mean().item()
            if best_inliers > 1:
                best_points[joint] = best_point.cpu()
        if torch.all(best_points != -1):
            hand_joints_3d[hand] = best_points

    hand_joints_3d = hand_joints_3d.numpy().astype(np.float32)

    return hand_joints_3d


def runner_draw_handmarks_results(rgb_images, handmarks, serials):
    vis_image = display_images(
        images=[
            draw_debug_image(
                rgb_image,
                hand_marks=handmarks[:, idx],
                draw_boxes=True,
                draw_hand_sides=True,
            )
            for idx, rgb_image in enumerate(rgb_images)
        ],
        names=serials,
        return_array=True,
    )
    return vis_image


def complete_3d_joints_by_cubic(joints_3d, ratio=0.5):
    def calculate_bone_lengths(joints_3d):
        """
        Calculate the bone lengths from the parent-child joint relationships.

        Parameters:
        - joints_3d: Observed 3D joints, shape (N, 21, 3).

        Returns:
        - Bone lengths for each joint, shape (21,).
        """
        bone_lengths = np.zeros(21)
        for i in range(1, 21):  # Skip the root joint, which has no parent
            parent_idx = HAND_JOINT_PARENTS[i]
            if parent_idx >= 0:  # Valid parent index
                bone_lengths[i] = np.linalg.norm(
                    joints_3d[:, parent_idx] - joints_3d[:, i], axis=1
                ).mean()
        return bone_lengths

    def adjust_joints_to_bone_lengths(joints_3d, bone_lengths):
        """
        Adjust the positions of the joints to respect the given bone lengths.

        Parameters:
        - joints_3d: Interpolated 3D joints, shape (N, 21, 3).
        - bone_lengths: Bone lengths to enforce, shape (21,).
        """
        for i in range(1, 21):  # Skip the root joint
            parent_idx = HAND_JOINT_PARENTS[i]
            if parent_idx >= 0:  # Valid parent index
                direction = joints_3d[:, i] - joints_3d[:, parent_idx]
                # Normalize the direction vector
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                joints_3d[:, i] = joints_3d[:, parent_idx] + direction * bone_lengths[i]

    hands, N, joints, coords = joints_3d.shape
    complete_joints = joints_3d.copy()
    for hand in range(hands):
        valid_frames = np.where(np.all(complete_joints[hand] != -1, axis=(1, 2)))[0]

        if len(valid_frames) < int(N * ratio):
            print(
                f"** Not enough valid frames for interpolation for hand-{hand}. (#frames: {len(valid_frames)}/{N})"
            )
            continue

        # Calculate average bone lengths from the first frame (assuming it's fully observed)
        bone_lengths = calculate_bone_lengths(joints_3d[hand, valid_frames])

        for joint in range(joints):
            for coord in range(coords):
                valid_coords = complete_joints[hand, valid_frames, joint, coord]
                cs = CubicSpline(valid_frames, valid_coords, bc_type="clamped")
                interpolated_coords = cs(np.arange(N))
                complete_joints[hand, :, joint, coord] = interpolated_coords
        # Adjust the completed joints to respect bone lengths
        adjust_joints_to_bone_lengths(complete_joints[hand, :], bone_lengths)

    return complete_joints


def complete_3d_joints_by_linear(joints_3d):
    hands, N, joints, coords = joints_3d.shape
    # Loop over each hand and each joint
    for hand in range(hands):
        if np.all(joints_3d[hand] == -1):  # no hand detected
            continue
        for joint in range(joints):
            for coord in range(coords):
                # Extract the current sequence for the joint's coordinate
                sequence = joints_3d[hand, :, joint, coord]
                # Identify frames where the joint data is not missing
                valid_frames = np.where(sequence != -1)[0]
                # Check if there are enough points to interpolate
                if len(valid_frames) > 1:
                    # Extract the valid coordinates and corresponding frames
                    valid_coords = sequence[valid_frames]
                    # Create a spline interpolation function
                    interp_func = interp1d(
                        valid_frames,
                        valid_coords,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(valid_coords[0], valid_coords[-1]),
                    )
                    # Interpolate missing points
                    interpolated_coords = interp_func(np.arange(N))
                    # Update the original array with interpolated values
                    joints_3d[hand, :, joint, coord] = interpolated_coords
    return joints_3d


class ManoPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder).resolve()
        self._logger = get_logger(
            log_level="DEBUG" if debug else "INFO", log_name="ManoPoseSolver"
        )
        self._save_folder = self._data_folder / "processed" / "hand_detection"
        self._save_folder.mkdir(parents=True, exist_ok=True)
        self._vis_folder = self._save_folder / "vis"
        self._vis_folder.mkdir(parents=True, exist_ok=True)

        # load variables from sequence loader
        self._loader = SequenceLoader(sequence_folder)
        self._serials = self._loader.serials
        self._master = self._loader.master_serial
        self._num_frames = self._loader.num_frames
        self._width = self._loader.width
        self._height = self._loader.height
        self._mano_sides = self._loader.mano_sides
        self._M = self._loader.M2world.cpu().numpy()
        self._num_cameras = len(self._serials)

    def run(self):
        # Run RANSAC 3D hand joints estimation
        self._run_hand_joints_3d_estimation()

    def _run_hand_joints_3d_estimation(self):
        self._logger.info(">>>>>>>>>> Running Hand 3D Joints Estimation <<<<<<<<<<")
        mp_handmarks_file = self._save_folder / "mp_handmarks_results.npz"
        if not mp_handmarks_file.exists():
            self._logger.error(
                "    ** Hand 2D Joints Detection Results does not exist!!!"
            )
            return

        joints_3d_file = self._save_folder / "hand_joints_3d_raw.npy"
        if joints_3d_file.exists():
            self._logger.info(
                "    ** Hand 3D Joints Estimation Results already exist!!!"
            )
            hand_joints_3d = np.load(joints_3d_file)
        else:
            # load mp_handmarks
            mp_handmarks = np.load(mp_handmarks_file)
            mp_handmarks = np.stack(
                [mp_handmarks[s] for s in self._serials], axis=2
            )  # (num_hands, num_frames, num_cameras, num_joints, 2)
            tqdm.write(f"mp_handmarks: {mp_handmarks.shape}")

            self._logger.info("    ** Running hand 3D joints RANSAC...")

            hand_joints_3d = [None] * self._num_frames
            tqbar = tqdm(total=self._num_frames, ncols=60)
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        runner_hand_3d_joints_ransac_torch,
                        mp_handmarks[:, frame_id],
                        self._M,
                        threshold=10.0,
                        iterations=500,
                        lr=1e-3,
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }

                for future in concurrent.futures.as_completed(futures):
                    frame_id = futures[future]
                    hand_joints_3d[frame_id] = future.result()
                    tqbar.update()
                    tqbar.refresh()

                del futures
            tqbar.close()

            hand_joints_3d = np.stack(
                hand_joints_3d, axis=1
            )  # (num_hands, num_frames, num_joints, 3)

            self._logger.info("    ** Saving Hand 3D Joints Estimation Results...")
            np.save(joints_3d_file, hand_joints_3d)

        # hand_joints_3d = complete_3d_joints_by_cubic(hand_joints_3d, ratio=0.8)
        hand_joints_3d = complete_3d_joints_by_linear(hand_joints_3d)

        hand_joints_2d = np.stack(
            [
                [
                    self._points_3d_to_2d(hand_joints_3d[hand_ind, frame_id], self._M)
                    for frame_id in range(self._num_frames)
                ]
                for hand_ind in range(2)
            ],
            axis=0,
        ).astype(np.int64)
        tqdm.write(f"hand_joints_2d: {hand_joints_2d.shape}")

        hand_joints_bbox = np.stack(
            [
                [
                    [
                        get_bbox_from_landmarks(pts_2d, self._width, self._height, 10)
                        for pts_2d in hand_joints_2d[hand_ind, frame_id]
                    ]
                    for frame_id in range(self._num_frames)
                ]
                for hand_ind in range(2)
            ],
            axis=0,
        ).astype(np.int64)
        tqdm.write(f"hand_joints_bbox: {hand_joints_bbox.shape}")

        # save 2D hand joints
        np.savez_compressed(
            self._save_folder / "hand_joints_3d_projection.npz",
            **{
                self._serials[cam_idx]: hand_joints_2d[:, :, cam_idx]
                for cam_idx in range(self._num_cameras)
            },
        )

        # save 2D hand joints bbox
        np.savez_compressed(
            self._save_folder / "hand_joints_3d_bbox.npz",
            **{
                self._serials[cam_idx]: hand_joints_bbox[:, :, cam_idx]
                for cam_idx in range(self._num_cameras)
            },
        )

        # draw visualization results
        tqdm.write("Drawing visualization results...")
        tqdm.write("    ** Generating vis images...")
        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=60, colour="green")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            tqdm.write("    ** Generating vis images for hand joints 3D...")
            futures = {
                executor.submit(
                    runner_draw_handmarks_results,
                    self._loader.get_rgb_image(frame_id),
                    hand_joints_2d[:, frame_id],
                    self._serials,
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                vis_images[futures[future]] = future.result()
                tqbar.update()
                tqbar.refresh()
            del futures
        tqbar.close()

        tqdm.write("    ** Saving vis images...")
        save_vis_folder = self._save_folder / "vis" / "hand_joints_3d"
        save_vis_folder.mkdir(parents=True, exist_ok=True)
        tqbar = tqdm(total=self._num_frames, ncols=60, colour="green")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_vis_folder / f"vis_{frame_id:06d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update()
                tqbar.refresh()
            del futures
        tqbar.close()

        tqdm.write("    ** Saving vis video...")
        create_video_from_rgb_images(
            self._save_folder / "vis" / "hand_joints_3d.mp4",
            vis_images,
            fps=30,
        )

        self._logger.info("Hand joints 3D estimation done.")

    def _point_3d_to_2d(self, point_3d, M):
        point_2d = M @ np.append(point_3d, 1)
        point_2d /= point_2d[2]
        point_2d = point_2d[:2].astype(np.int64)
        # check if the point is within the image
        if (
            point_2d[0] < 0
            or point_2d[0] >= self._width
            or point_2d[1] < 0
            or point_2d[1] >= self._height
        ):
            return None
        return point_2d

    def _points_3d_to_2d(self, points_3d, Ms):
        points_2d = np.full((len(Ms), len(points_3d), 2), -1)
        for i, p_3d in enumerate(points_3d):
            if np.any(p_3d == -1):
                continue
            tmp_pts_2d = [self._point_3d_to_2d(p_3d, M) for M in Ms]
            # replace invalid points with -1
            points_2d[:, i] = [
                p_2d if p_2d is not None else np.array([-1, -1]) for p_2d in tmp_pts_2d
            ]
        return points_2d.astype(np.int64)


def args_parser():
    parser = argparse.ArgumentParser(description="Hand Detection")
    parser.add_argument(
        "--sequence_folder",
        required=True,
        help="sequence folder",
        type=str,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    sequence_folder = args.sequence_folder
    debug = args.debug

    pose_estimator = ManoPoseSolver(sequence_folder=sequence_folder, debug=debug)
    pose_estimator.run()
