import torch
from .Utils import *


class SequenceLoader:
    def __init__(
        self,
        sequence_folder,
        load_mano=False,
        load_object=False,
        in_world=True,
        device="cpu",
        debug=False,
    ) -> None:
        self._data_folder = Path(sequence_folder).resolve()
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._model_folder = self._data_folder.parent.parent / "models"
        self._device = device
        self._load_mano = load_mano
        self._load_object = load_object
        self._in_world = in_world

        self._logger = get_logger(
            log_level="DEBUG" if debug else "INFO", log_name="SequenceLoader"
        )

        # Crop limits in world frame, [x_min, x_max, y_min, y_max, z_min, z_max]
        self._crop_lim = [-0.60, +0.60, -0.40, +0.40, -0.01, +0.80]

        # Load metadata
        self._load_metadata()

        # Create 3D rays
        self._rays = self._create_3d_rays()

        # Projection matrix
        self._M2master = torch.bmm(self._Ks, self._extr2master_inv[:, :3, :])
        self._M2world = torch.bmm(self._Ks, self._extr2world_inv[:, :3, :])

        # Initialize variables
        self._frame_id = -1
        self._pcd_points = torch.zeros(
            (self._num_cams, self._height * self._width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._pcd_colors = torch.zeros(
            (self._num_cams, self._height * self._width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._pcd_masks = torch.zeros(
            (self._num_cams, self._height * self._width),
            dtype=torch.bool,
            device=self._device,
        )

    def _load_metadata(self):
        data = read_data_from_json(self._data_folder / "meta.json")
        self._serials = data.get("serials", None)
        self._width = data.get("width", None)
        self._height = data.get("height", None)
        self._subject_id = data.get("subject_id", None)
        self._object_ids = data.get("object_ids", None)
        self._num_frames = data.get("num_frames", None)
        self._mano_sides = data.get("mano_sides", None)
        self._num_cams = len(self._serials)

        self._load_intrinsics(data["serials"])
        self._load_extrinsics(
            self._calib_folder / "extrinsics" / data["extrinsics"] / "extrinsics.json"
        )
        self._load_mano_beta(
            self._calib_folder / "mano" / data["mano_calib"] / "mano.json"
        )

    def _load_intrinsics(self, serials):
        def load_K_from_json(json_file, cam_type="color"):
            data = read_data_from_json(json_file)
            K = np.array(
                [
                    [data[cam_type]["fx"], 0, data[cam_type]["ppx"]],
                    [0, data[cam_type]["fy"], data[cam_type]["ppy"]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            return K

        Ks = np.stack(
            [
                load_K_from_json(
                    self._calib_folder
                    / "intrinsics"
                    / f"{serial}_{self._width}x{self._height}.json"
                )
                for serial in serials
            ],
            axis=0,
        )

        self._Ks = torch.from_numpy(Ks).to(self._device)
        self._Ks_inv = torch.inverse(self._Ks)

    def _load_extrinsics(self, extr_file):
        data = read_data_from_json(extr_file)
        self._master = data["rs_master"]
        tag1 = np.array(
            [
                [
                    data["extrinsics"]["tag_1"][0],
                    data["extrinsics"]["tag_1"][1],
                    data["extrinsics"]["tag_1"][2],
                    data["extrinsics"]["tag_1"][3],
                ],
                [
                    data["extrinsics"]["tag_1"][4],
                    data["extrinsics"]["tag_1"][5],
                    data["extrinsics"]["tag_1"][6],
                    data["extrinsics"]["tag_1"][7],
                ],
                [
                    data["extrinsics"]["tag_1"][8],
                    data["extrinsics"]["tag_1"][9],
                    data["extrinsics"]["tag_1"][10],
                    data["extrinsics"]["tag_1"][11],
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        tag1_inv = np.linalg.inv(tag1)
        extr2master = np.stack(
            [
                np.array(
                    [
                        [
                            data["extrinsics"][serial][0],
                            data["extrinsics"][serial][1],
                            data["extrinsics"][serial][2],
                            data["extrinsics"][serial][3],
                        ],
                        [
                            data["extrinsics"][serial][4],
                            data["extrinsics"][serial][5],
                            data["extrinsics"][serial][6],
                            data["extrinsics"][serial][7],
                        ],
                        [
                            data["extrinsics"][serial][8],
                            data["extrinsics"][serial][9],
                            data["extrinsics"][serial][10],
                            data["extrinsics"][serial][11],
                        ],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                for serial in self._serials
            ],
            axis=0,
        )
        extr2master_inv = np.stack([np.linalg.inv(T) for T in extr2master], axis=0)
        extr2world = np.stack([np.dot(tag1_inv, T) for T in extr2master], axis=0)
        extr2world_inv = np.stack([np.linalg.inv(T) for T in extr2world], axis=0)

        self._extr2master = torch.from_numpy(extr2master).to(self._device)
        self._extr2master_inv = torch.from_numpy(extr2master_inv).to(self._device)
        self._extr2world = torch.from_numpy(extr2world).to(self._device)
        self._extr2world_inv = torch.from_numpy(extr2world_inv).to(self._device)

    def _load_mano_beta(self, beta_file):
        data = read_data_from_json(beta_file)
        self._mano_beta = torch.tensor(
            data["betas"], dtype=torch.float32, device=self._device
        )

    def _create_3d_rays(self):
        def create_2d_coords(H, W):
            xv, yv = torch.meshgrid(
                torch.arange(0, W), torch.arange(0, H), indexing="xy"
            )
            coords_2d = torch.stack(
                [xv, yv, torch.ones_like(xv)], dim=0
            ).float()  # 3 x H x W
            return coords_2d.to(self._device)

        coords_2d = create_2d_coords(self._height, self._width)
        coords_2d = (
            coords_2d.unsqueeze(0)
            .repeat(self._num_cams, 1, 1, 1)
            .view(self._num_cams, 3, -1)
        )  # N x 3 x H*W
        rays = torch.bmm(self._Ks_inv, coords_2d)
        return rays

    def deproject(self, colors, depths, depth_scale=1000.0):
        # Process colors
        colors = (
            torch.from_numpy(
                np.stack(colors, axis=0, dtype=np.float32).reshape(
                    self._num_cams, -1, 3
                )
            ).to(self._device)
            / 255.0
        )  # N x H*W x 3

        # Process depths
        depths = (
            torch.from_numpy(
                np.stack(depths, axis=0, dtype=np.float32).reshape(
                    self._num_cams, 1, -1
                )
            ).to(self._device)
            / depth_scale
        )  # N x 1 x H*W

        # Deproject to 3D points in camera frame
        pts_c = self._rays * depths  # N x 3 x H*W

        # transform to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # N x H*W x 3

        # Filter points outside crop limits
        mx1 = pts[..., 0] > self._crop_lim[0]
        mx2 = pts[..., 0] < self._crop_lim[1]
        my1 = pts[..., 1] > self._crop_lim[2]
        my2 = pts[..., 1] < self._crop_lim[3]
        mz1 = pts[..., 2] > self._crop_lim[4]
        mz2 = pts[..., 2] < self._crop_lim[5]
        mask = mx1 & mx2 & my1 & my2 & mz1 & mz2

        # Transform to master frame if needed
        if not self._in_world:
            pts = torch.baddbmm(
                self._extr2master[:, :3, 3].unsqueeze(2),
                self._extr2master[:, :3, :3],
                pts_c,
            ).permute(
                0, 2, 1
            )  # N x H*W x 3

        return colors, pts, mask

    def _update_pcd(self, frame_id):
        colors, points, masks = self.deproject(
            self.get_rgb_image(frame_id),
            self.get_depth_image(frame_id),
        )
        self._pcd_points.copy_(points)
        self._pcd_colors.copy_(colors)
        self._pcd_masks.copy_(masks)

    def get_rgb_image(self, frame_id, serial=None):
        if serial is None:
            data = [None] * self._num_cams
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        read_rgb_image,
                        self._data_folder / serial / f"color_{frame_id:06d}.jpg",
                    ): i
                    for i, serial in enumerate(self._serials)
                }
                for future in concurrent.futures.as_completed(futures):
                    data[futures[future]] = future.result()
        else:
            data = read_rgb_image(
                self._data_folder / serial / f"color_{frame_id:06d}.jpg"
            )
        return data

    def get_depth_image(self, frame_id, serial=None):
        if serial is None:
            data = [None] * self._num_cams
            with concurrent.futures.ThreadPoolExecutor() as executor:
                workers = [
                    executor.submit(
                        read_depth_image,
                        self._data_folder / s / f"depth_{frame_id:06d}.png",
                        idx=i,
                    )
                    for i, s in enumerate(self._serials)
                ]
                for worker in concurrent.futures.as_completed(workers):
                    img, idx = worker.result()
                    data[idx] = img
        else:
            data = read_depth_image(
                self._data_folder / serial / f"depth_{frame_id:06d}.png"
            )
        return data

    def get_mask_image(self, frame_id, serial=None):
        if serial is None:
            data = [None] * self._num_cams
            workers = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, s in enumerate(self._serials):
                    mask_file = self._data_folder / s / f"mask_{frame_id:06d}.png"
                    if mask_file.exists():
                        workers.append(
                            executor.submit(read_mask_image, mask_file, idx=i)
                        )
                    else:
                        data[i] = np.zeros((self._height, self._width), dtype=np.uint8)
                for worker in workers:
                    img, idx = worker.result()
                    data[idx] = img
        else:
            mask_file = self._data_folder / serial / f"mask_{frame_id:06d}.png"
            if mask_file.exists():
                data = read_mask_image(mask_file)
            else:
                data = np.zeros((self._height, self._width), dtype=np.uint8)

        return data

    def step(self):
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id):
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    @property
    def Ks(self):
        return self._Ks

    @property
    def Ks_inv(self):
        return self._Ks_inv

    @property
    def extrinsics2master(self):
        return self._extr2master

    @property
    def extrinsics2master_inv(self):
        return self._extr2master_inv

    @property
    def extrinsics2world(self):
        return self._extr2world

    @property
    def extrinsics2world_inv(self):
        return self._extr2world_inv

    @property
    def M2master(self):
        """Projection matrix from 3D points to master frame."""
        return self._M2master

    @property
    def M2world(self):
        """Projection matrix from 3D points to world frame."""
        return self._M2world

    @property
    def mano_beta(self):
        return self._mano_beta

    @property
    def device(self):
        return self._device

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def serials(self):
        return self._serials

    @property
    def master_serial(self):
        return self._master

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def mano_sides(self):
        return self._mano_sides

    @property
    def object_textured_mesh_files(self):
        return (
            [
                self._calib_folder / "objects" / oid / f"textured_mesh.obj"
                for oid in self._object_ids
            ]
            if self._load_object
            else None
        )

    @property
    def object_cleaned_mesh_files(self):
        return (
            [
                self._calib_folder / "objects" / oid / f"cleaned_mesh_10000.obj"
                for oid in self._object_ids
            ]
            if self._load_object
            else None
        )

    @property
    def pcd_points(self):
        return self._pcd_points

    @property
    def pcd_colors(self):
        return self._pcd_colors

    @property
    def pcd_masks(self):
        return self._pcd_masks

    @property
    def pcd_points_map(self):
        return self._pcd_points.view(self._num_cams, self._height, self._width, 3)

    @property
    def pcd_colors_map(self):
        return self._pcd_colors.view(self._num_cams, self._height, self._width, 3)

    @property
    def pcd_masks_map(self):
        return self._pcd_masks.view(self._num_cams, self._height, self._width)
