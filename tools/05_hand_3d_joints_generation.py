import argparse
import itertools

from _init_paths import *
from lib.Utils import *
from lib.SequenceLoader import SequenceLoader


# Set the start method to 'spawn'
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class ManoPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._debug = debug
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
        camera_pairs = list(itertools.combinations(range(self._num_cameras), 2))

        hand_joints_3d = np.full((2, self._num_frames, 21, 3), -1, dtype=np.float32)
        hand_joints_3d_projected = np.full(
            (2, self._num_frames, self._num_cameras, 21, 2), -1, dtype=np.int64
        )
        for frame_id in tqdm(range(self._num_frames), ncols=60, colour="green"):
            # load 2D handmarks from mediapipe hand detection
            mp_handmarks = np.stack(
                [
                    np.load(
                        self._hand_detection_folder
                        / serial
                        / f"handmarks_{frame_id:06d}.npy"
                    )
                    for serial in self._serials
                ],
                dtype=np.float32,
                axis=1,
            )  # (num_hands, num_cameras, num_joints, 2)

            # create candidate 3d hand joints by triangulating each pair of 2D handmarks
            for mano_side in self._mano_sides:
                hand_ind = 0 if mano_side == "right" else 1
                handmarks = mp_handmarks[hand_ind]

                if np.all(handmarks == -1):
                    self._logger.warning(
                        f"Frame {frame_id:06d}: No hand detected for {mano_side}."
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

                best_pts_3d = np.stack(best_pts_3d, axis=0, dtype=np.float32)
                best_pts_2d = self._points_3d_to_2d(best_pts_3d, self._M)

                # save 3D hand joints
                hand_joints_3d[hand_ind, frame_id] = best_pts_3d
                hand_joints_3d_projected[hand_ind, frame_id] = best_pts_2d.astype(int)

        # save 3D hand joints
        np.save(
            self._hand_detection_folder / "hand_joints_3d.npy", hand_joints_3d
        )  # (num_hands, num_frames, num_joints, 3)

        # save 2D hand joints
        np.save(
            self._hand_detection_folder / "hand_joints_3d_projection.npy",
            hand_joints_3d_projected,
        )

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

        # for combination in self._get_all_combinations(len(pts_3d)):
        for pt_3d in pts_3d:
            # pts = pts_3d[list(combination)]
            # pt_3d = np.mean(pts, axis=0)
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
    # args = args_parser()
    # sequence_folder = args.sequence_folder
    # debug = args.debug

    sequence_folder = PROJ_ROOT / "data/recordings/ida_20240617_101133"
    debug = True

    estimator = ManoPoseSolver(sequence_folder=sequence_folder, debug=debug)
    estimator.run_hand_joints_3d_estimation()
