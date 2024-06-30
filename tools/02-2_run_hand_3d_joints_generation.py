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
        self._hand_detection_folder = self._data_folder / "processed" / "hand_detection"
        self._hand_detection_folder.mkdir(parents=True, exist_ok=True)

        # load variables from sequence loader
        self._loader = SequenceLoader(sequence_folder)
        self._serials = self._loader.serials
        self._master_serial = self._loader.master_serial
        self._num_frames = self._loader.num_frames
        self._width = self._loader.width
        self._height = self._loader.height
        self._mano_sides = self._loader.mano_sides
        self._M = self._loader.M2world.cpu().numpy()
        self._num_cameras = len(self._serials)

    def run_hand_joints_3d_estimation(self):
        self._logger.info("Start hand joints 3D estimation...")
        # load mp handmarks
        mp_handmarks = np.load(self._hand_detection_folder / "mp_handmarks_results.npz")
        mp_handmarks = np.stack(
            [mp_handmarks[serial] for serial in self._serials], axis=2
        )  # (num_hands, num_frames, num_cameras, num_joints, 2)

        camera_pairs = list(itertools.combinations(range(self._num_cameras), 2))
        hand_joints_3d = np.full((2, self._num_frames, 21, 3), -1, dtype=np.float32)
        for frame_id in tqdm(range(self._num_frames), ncols=60, colour="green"):
            # create candidate 3d hand joints by triangulating each pair of 2D handmarks
            for mano_side in self._mano_sides:
                hand_ind = 0 if mano_side == "right" else 1
                handmarks = mp_handmarks[hand_ind, frame_id]

                num_valid_cameras = np.sum(np.all(handmarks != -1, axis=(1, 2)))
                if num_valid_cameras < 4:
                    self._logger.warning(
                        f"Frame {frame_id:06d}: Less than 4 cameras detected for {mano_side}."
                    )
                    continue

                best_pts_3d = []

                for jt_idx in range(21):
                    marks = handmarks[:, jt_idx]
                    pts_3d = []
                    for cam_pair in camera_pairs:
                        cam_idx_1, cam_idx_2 = cam_pair
                        mark1 = marks[cam_idx_1]
                        mark2 = marks[cam_idx_2]
                        if np.any(mark1 == -1) or np.any(mark2 == -1):
                            continue
                        pt_3d = self._triangulate_handmarks(
                            mark1, mark2, cam_idx_1, cam_idx_2
                        )
                        pts_3d.append(pt_3d)

                    pts_3d = np.stack(pts_3d)

                    # optimize the 3D points using RANSAC
                    best_pt_3d = self._get_best_3d_point_ransac(
                        pts_3d, marks, thresh=5.0
                    )
                    best_pts_3d.append(best_pt_3d)

                best_pts_3d = np.stack(
                    best_pts_3d, axis=0, dtype=np.float32
                )  # (num_joints, 3)

                # save 3D hand joints
                hand_joints_3d[hand_ind, frame_id] = best_pts_3d

        hand_joints_3d = hand_joints_3d.astype(np.float32)

        # complete 3D hand joints by interpolation
        # hand_joints_3d = complete_3d_joints_by_cubic(hand_joints_3d)
        hand_joints_3d = complete_3d_joints_by_linear(hand_joints_3d)

        # generate 2D hand joints projection for each camera
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

        # generate 2D hand joints bbox for each camera
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

        # save 3D hand joints
        np.save(self._hand_detection_folder / "hand_joints_3d.npy", hand_joints_3d)

        # save 2D hand joints
        np.savez_compressed(
            self._hand_detection_folder / "hand_joints_3d_projection.npz",
            **{
                self._serials[cam_idx]: hand_joints_2d[:, :, cam_idx]
                for cam_idx in range(self._num_cameras)
            },
        )

        # save 2D hand joints bbox
        np.savez_compressed(
            self._hand_detection_folder / "hand_joints_3d_bbox.npz",
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
        save_vis_folder = self._hand_detection_folder / "vis" / "hand_joints_3d"
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
            self._hand_detection_folder / "vis" / "hand_joints_3d.mp4",
            vis_images,
            fps=30,
        )

        self._logger.info("Hand joints 3D estimation done.")

    def _triangulate_handmarks(self, mark1, mark2, cam_idx_1, cam_idx_2):
        M1 = self._M[cam_idx_1]
        M2 = self._M[cam_idx_2]
        A = np.zeros((4, 4))
        A[0] = mark1[0] * M1[2] - M1[0]
        A[1] = mark1[1] * M1[2] - M1[1]
        A[2] = mark2[0] * M2[2] - M2[0]
        A[3] = mark2[1] * M2[2] - M2[1]

        U, S, Vh = np.linalg.svd(A, full_matrices=False)

        X = Vh[-1, :3] / Vh[-1, 3]
        return X

    def _get_all_combinations(self, n):
        combinations = []
        for i in range(1, n + 1):
            combinations.extend(list(itertools.combinations(range(n), i)))
        return combinations

    def _get_best_3d_point_ransac(self, pts_3d, marks, thresh=10.0):
        best_pt_3d = np.array([-1, -1, -1], dtype=np.float32)
        best_inliers = 0
        best_loss = 0.0

        for pt_3d in pts_3d:
            loss = self._projection_loss(pt_3d, marks)
            inliers = np.sum(loss < thresh)
            if inliers > best_inliers:
                best_inliers = inliers
                best_pt_3d = pt_3d
                best_loss = np.mean(loss)
            if (
                best_inliers > 0
                and inliers == best_inliers
                and np.mean(loss) < best_loss
            ):
                best_inliers = inliers
                best_pt_3d = pt_3d
                best_loss = np.mean(loss)
        return best_pt_3d

    def _projection_loss(self, p_3d, marks):
        valid_cam_inds = np.where(np.all(marks != -1, axis=1))[0]
        proj_pts = np.stack(
            [self._point_3d_to_2d(p_3d, self._M[cam_idx]) for cam_idx in valid_cam_inds]
        )
        proj_valid_cam_inds = np.where(np.all(proj_pts != -1, axis=1))[0]
        X = marks[valid_cam_inds][proj_valid_cam_inds]
        Y = proj_pts[proj_valid_cam_inds]

        # Calculate the mean of the squared differences
        # loss = np.mean(np.square(X - Y), axis=1)  # MSE
        loss = np.mean(np.abs(X - Y), axis=1)  # MAE
        return loss

    def _point_3d_to_2d(self, point_3d, M):
        point_2d = M @ np.append(point_3d, 1)
        point_2d /= point_2d[2]
        point_2d = point_2d[:2]
        # check if the point is within the image
        if (
            point_2d[0] < 0
            or point_2d[0] >= self._width
            or point_2d[1] < 0
            or point_2d[1] >= self._height
        ):
            point_2d = np.array([-1, -1])
        return point_2d.astype(np.float32)

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
        return points_2d.astype(np.float32)


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

    estimator = ManoPoseSolver(sequence_folder=sequence_folder, debug=debug)
    estimator.run_hand_joints_3d_estimation()
