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
        self._models_folder = self._data_folder.parent.parent / "models"
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
        self._create_3d_rays()

        # Projection matrix
        self._M2master = torch.bmm(self._Ks, self._extr2master_inv[:, :3, :])
        self._M2world = torch.bmm(self._Ks, self._extr2world_inv[:, :3, :])

        # Initialize MANO and Object group layers
        self._initialize_mano_group_layer()
        self._initialize_object_group_layer()

        # Initialize variables
        self._frame_id = -1
        self._initialize_pcd_variables()

    def _load_metadata(self):
        """Load metadata, intrinsics, extrinsics, and mano beta."""

        data = read_data_from_json(self._data_folder / "meta.json")

        self._serials = data.get("serials")
        self._width = data.get("width")
        self._height = data.get("height")
        self._subject_id = data.get("mano_calib")
        self._object_id = data.get("object_ids")
        self._num_frames = data.get("num_frames")
        self._mano_sides = data.get("mano_sides")
        self._num_cams = len(self._serials)

        self._load_intrinsics(self._serials)
        self._load_extrinsics(
            self._calib_folder / "extrinsics" / data["extrinsics"] / "extrinsics.json"
        )
        self._load_mano_beta(
            self._calib_folder / "mano" / self._subject_id / "mano.json"
        )

    def _load_intrinsics(self, serials):
        def load_K_from_json(json_file, cam_type="color"):
            data = read_data_from_json(json_file)
            return np.array(
                [
                    [data[cam_type]["fx"], 0, data[cam_type]["ppx"]],
                    [0, data[cam_type]["fy"], data[cam_type]["ppy"]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

        intrinsics_path = self._calib_folder / "intrinsics"
        Ks = np.array(
            [
                load_K_from_json(
                    intrinsics_path / f"{serial}_{self._width}x{self._height}.json"
                )
                for serial in serials
            ],
            dtype=np.float32,
        )

        Ks_inv = np.linalg.inv(Ks)

        self._Ks = torch.from_numpy(Ks).to(self._device)
        self._Ks_inv = torch.from_numpy(Ks_inv).to(self._device)

    def _load_extrinsics(self, extr_file):
        def extract_matrix(data, key):
            extr = data["extrinsics"][key]
            matrix = np.array(
                [
                    [extr[0], extr[1], extr[2], extr[3]],
                    [extr[4], extr[5], extr[6], extr[7]],
                    [extr[8], extr[9], extr[10], extr[11]],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            return matrix

        data = read_data_from_json(extr_file)
        self._master = data["rs_master"]
        tag1 = extract_matrix(data, "tag_1")
        tag1_inv = np.linalg.inv(tag1)

        extr2master = np.array(
            [extract_matrix(data, serial) for serial in self._serials], dtype=np.float32
        )
        extr2master_inv = np.linalg.inv(extr2master)
        extr2world = np.array([np.dot(tag1_inv, extr) for extr in extr2master])
        extr2world_inv = np.linalg.inv(extr2world)

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
        """Create 3D rays."""

        def create_2d_coords(height, width):
            """Create a 2D grid of coordinates."""
            xv, yv = torch.meshgrid(
                torch.arange(0, width, device=self._device),
                torch.arange(0, height, device=self._device),
                indexing="xy",
            )
            coords_2d = torch.stack(
                [xv, yv, torch.ones_like(xv)], dim=0
            ).float()  # 3 x H x W
            return coords_2d

        # Create 2D coordinates for the image dimensions
        coords_2d = create_2d_coords(self._height, self._width)

        # Expand and reshape coordinates for each camera
        coords_2d = coords_2d.unsqueeze(0).repeat(
            self._num_cams, 1, 1, 1
        )  # N x 3 x H x W
        coords_2d = coords_2d.view(self._num_cams, 3, -1)  # N x 3 x H*W

        # Compute 3D rays by multiplying with the inverse intrinsic matrix
        self._rays = torch.bmm(self._Ks_inv, coords_2d)  # N x 3 x H*W

    def _initialize_pcd_variables(self):
        """Initialize point cloud variables."""
        num_points = self._height * self._width
        self._pcd_points = torch.zeros(
            (self._num_cams, num_points, 3), dtype=torch.float32, device=self._device
        )
        self._pcd_colors = torch.zeros(
            (self._num_cams, num_points, 3), dtype=torch.float32, device=self._device
        )
        self._pcd_masks = torch.zeros(
            (self._num_cams, num_points), dtype=torch.bool, device=self._device
        )

    def _deproject(self, colors, depths, depth_scale=1000.0):
        # Process colors
        colors = (
            torch.from_numpy(
                np.stack(colors, axis=0)
                .reshape(self._num_cams, -1, 3)
                .astype(np.float32)
            ).to(self._device)
            / 255.0
        )  # N x H*W x 3

        # Process depths
        depths = (
            torch.from_numpy(
                np.stack(depths, axis=0)
                .reshape(self._num_cams, 1, -1)
                .astype(np.float32)
            ).to(self._device)
            / depth_scale
        )  # N x 1 x H*W

        # Deproject to 3D points in camera frame
        pts_c = self._rays * depths  # N x 3 x H*W

        # Transform to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # N x H*W x 3

        # Filter points outside crop limits
        mask = (
            (pts[..., 0] > self._crop_lim[0])
            & (pts[..., 0] < self._crop_lim[1])
            & (pts[..., 1] > self._crop_lim[2])
            & (pts[..., 1] < self._crop_lim[3])
            & (pts[..., 2] > self._crop_lim[4])
            & (pts[..., 2] < self._crop_lim[5])
        )

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
        # Retrieve RGB and depth images for the given frame
        rgb_images = self.get_rgb_image(frame_id)
        depth_images = self.get_depth_image(frame_id)

        # Deproject the images to get colors, points, and masks
        colors, points, masks = self._deproject(rgb_images, depth_images)

        # Update the point cloud data
        self._pcd_points.copy_(points)
        self._pcd_colors.copy_(colors)
        self._pcd_masks.copy_(masks)

    def _initialize_mano_group_layer(self):
        """Initialize the MANO group layer."""
        if not self._load_mano:
            self._mano_group_layer = None
        else:
            from .Layers import MANOGroupLayer

            mano_beta = self._mano_beta.cpu().numpy()
            self._mano_group_layer = MANOGroupLayer(
                self._mano_sides, [mano_beta for _ in self._mano_sides]
            ).to(self._device)

    def _initialize_object_group_layer(self):
        """Initialize the Object group layer."""
        if not self._load_object:
            self._object_group_layer = None
        else:
            from .Layers import ObjectGroupLayer

            obj_mesh = trimesh.load_mesh(self.object_cleaned_mesh_file, process=False)
            self._object_group_layer = ObjectGroupLayer(
                [obj_mesh.vertices], [obj_mesh.faces], [obj_mesh.vertex_normals]
            ).to(self._device)

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
        def read_mask_for_serial(serial):
            mask_file = self._data_folder / serial / f"mask_{frame_id:06d}.png"
            if mask_file.exists():
                mask = read_mask_image(mask_file)
                if mask is None:
                    mask = np.zeros((self._height, self._width), dtype=np.uint8)
            else:
                mask = np.zeros((self._height, self._width), dtype=np.uint8)
            return mask

        if serial is None:
            data = [None] * self._num_cams
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(read_mask_for_serial, serial): i
                    for i, serial in enumerate(self._serials)
                }
                for future in concurrent.futures.as_completed(futures):
                    data[futures[future]] = future.result()
        else:
            data = read_mask_for_serial(serial)

        return data

    def step(self):
        """Advances to the next frame, updating the point cloud data."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id):
        """Advances to a specific frame, updating the point cloud data."""
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    @property
    def Ks(self):
        """Intrinsic camera matrix for each camera."""
        return self._Ks

    @property
    def Ks_inv(self):
        """Inverse of the intrinsic camera matrix for each camera."""
        return self._Ks_inv

    @property
    def extrinsics2master(self):
        """Extrinsic transformation matrices from each camera to the master camera frame."""
        return self._extr2master

    @property
    def extrinsics2master_inv(self):
        """Inverse of the extrinsic transformation matrices from each camera to the master camera frame."""
        return self._extr2master_inv

    @property
    def extrinsics2world(self):
        """Extrinsic transformation matrices from each camera to the world frame."""
        return self._extr2world

    @property
    def extrinsics2world_inv(self):
        """Inverse of the extrinsic transformation matrices from each camera to the world frame."""
        return self._extr2world_inv

    @property
    def M2master(self):
        """Projection matrix from 3D points to the master camera frame."""
        return self._M2master

    @property
    def M2world(self):
        """Projection matrix from 3D points to the world frame."""
        return self._M2world

    @property
    def mano_beta(self):
        """MANO beta parameters."""
        return self._mano_beta

    @property
    def device(self):
        """Device on which tensors are allocated (e.g., 'cpu' or 'cuda')."""
        return self._device

    @property
    def num_frames(self):
        """Total number of frames in the sequence."""
        return self._num_frames

    @property
    def subject_id(self):
        """Identifier for the person being recorded."""
        return self._subject_id

    @property
    def width(self):
        """Width of the images."""
        return self._width

    @property
    def height(self):
        """Height of the images."""
        return self._height

    @property
    def serials(self):
        """List of camera serial numbers."""
        return self._serials

    @property
    def num_cameras(self):
        """Number of cameras used."""
        return self._num_cams

    @property
    def master_serial(self):
        """Serial number of the master camera."""
        return self._master

    @property
    def object_id(self):
        """The object ID being tracked."""
        return self._object_id

    @property
    def mano_sides(self):
        """List of sides (left/right) for MANO hand models."""
        return self._mano_sides

    @property
    def object_textured_mesh_file(self):
        """File path to the textured mesh file for the object."""
        return str(self._models_folder / self._object_id / "textured_mesh.obj")

    @property
    def object_cleaned_mesh_file(self):
        """File path to the cleaned mesh file for the object."""
        return str(self._models_folder / self._object_id / "cleaned_mesh_10000.obj")

    @property
    def pcd_points(self):
        """3D point cloud data points."""
        return self._pcd_points

    @property
    def pcd_colors(self):
        """Colors corresponding to the 3D point cloud data points."""
        return self._pcd_colors

    @property
    def pcd_masks(self):
        """Masks corresponding to the 3D point cloud data points."""
        return self._pcd_masks

    @property
    def pcd_points_map(self):
        """3D point cloud data points reshaped into a map."""
        return self._pcd_points.view(self._num_cams, self._height, self._width, 3)

    @property
    def pcd_colors_map(self):
        """Colors corresponding to the 3D point cloud data points reshaped into a map."""
        return self._pcd_colors.view(self._num_cams, self._height, self._width, 3)

    @property
    def pcd_masks_map(self):
        """Masks corresponding to the 3D point cloud data points reshaped into a map."""
        return self._pcd_masks.view(self._num_cams, self._height, self._width)

    @property
    def mano_group_layer(self):
        """MANO group layer."""
        return self._mano_group_layer

    @property
    def object_group_layer(self):
        """Object group layer."""
        return self._object_group_layer

    @property
    def mano_faces(self):
        """Faces of the MANO hand model (np.int64)."""
        if not self._load_mano:
            return None

        # Initialize the list with the original MANO faces
        mano_faces = [self._mano_group_layer.f.cpu().numpy()]

        # Append new faces for each side with appropriate vertex offset
        for i, side in enumerate(self._mano_sides):
            new_faces = np.array(
                NEW_MANO_FACES_RIGHT if side == "right" else NEW_MANO_FACES_LEFT
            )
            mano_faces.append(new_faces + i * NUM_MANO_VERTS)

        # Combine all faces into a single array
        mano_faces = np.vstack(mano_faces)

        return mano_faces

    @property
    def mano_colors(self):
        """Colors of the MANO hand model (float32)."""
        if not self._load_mano:
            return None

        mano_colors = []

        # Generate vertex colors for each side
        for side in self._mano_sides:
            # Select the appropriate color based on the side
            color = HAND_COLORS[1 if side == "right" else 2].rgb_norm
            # Repeat the color for each vertex
            side_colors = np.tile(color, (NUM_MANO_VERTS, 1))
            mano_colors.append(side_colors)

        # Combine all colors into a single array
        mano_colors = np.vstack(mano_colors).astype(np.float32)

        return mano_colors
