from pathlib import Path
import torch
from torch.nn import Module
from manopth.manolayer import ManoLayer

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent


class MANOLayer(Module):
    """Wrapper layer for manopath ManoLayer."""

    def __init__(self, side, betas):
        """
        Constructor.
        Args:
            side: MANO hand type. 'right' or 'left'.
            betas: A numpy array of shape [10] containing the betas.
        """
        super().__init__()
        self._side = side
        self._betas = betas

        self._mano_layer = ManoLayer(
            center_idx=0,
            flat_hand_mean=True,
            ncomps=45,
            side=side,
            mano_root=str(PROJ_ROOT / "config/mano_models"),
            use_pca=True,
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
            robust_rot=True,
        )

        # Register buffers
        self.register_buffer("b", torch.from_numpy(betas).unsqueeze(0).float())
        self.register_buffer("f", self._mano_layer.th_faces)
        self.register_buffer("root_trans", self._calculate_root_translation())

        # Initialize variables to store vertices and faces
        self._vertices = None
        self._faces = self.f

    def _calculate_root_translation(self):
        shapedirs = self._mano_layer.th_shapedirs
        v_template = self._mano_layer.th_v_template
        v = torch.matmul(shapedirs, self.b.t()).permute(2, 0, 1) + v_template
        return torch.matmul(self._mano_layer.th_J_regressor[0], v)

    def forward(self, p, t):
        """
        Forward function.
        Args:
            p: A tensor of shape [B, 48] containing the pose.
            t: A tensor of shape [B, 3] containing the trans.
        Returns:
            v: A tensor of shape [B, 778, 3] containing the vertices.
            j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)

        # Convert to meters.
        v /= 1000.0
        j /= 1000.0

        # Store vertices for normal calculation
        self._vertices = v

        return v, j

    @property
    def vertex_normals(self):
        """
        Calculate and return vertex normals.
        Returns:
            normals: A tensor of shape [B, 778, 3] containing the vertex normals.
        """
        if self._vertices is None:
            raise ValueError(
                "Vertices are not yet calculated. Run the forward method first."
            )

        return self.calculate_vertex_normals(self._vertices, self._faces)

    def calculate_vertex_normals(self, vertices, faces):
        """
        Calculate vertex normals from vertices and faces.
        Args:
            vertices: A tensor of shape [B, 778, 3] containing the vertices.
            faces: A tensor containing the face indices.
        Returns:
            normals: A tensor of shape [B, 778, 3] containing the vertex normals.
        """
        # Calculate face normals
        v0 = vertices[:, faces[:, 0], :]
        v1 = vertices[:, faces[:, 1], :]
        v2 = vertices[:, faces[:, 2], :]

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (
            torch.norm(face_normals, dim=2, keepdim=True) + 1e-10
        )

        # Initialize vertex normals
        vertex_normals = torch.zeros_like(vertices)

        # Sum face normals to vertex normals
        for i in range(faces.size(0)):
            vertex_normals[:, faces[i, 0], :] += face_normals[:, i, :]
            vertex_normals[:, faces[i, 1], :] += face_normals[:, i, :]
            vertex_normals[:, faces[i, 2], :] += face_normals[:, i, :]

        # Normalize vertex normals
        vertex_normals = vertex_normals / (
            torch.norm(vertex_normals, dim=2, keepdim=True) + 1e-10
        )

        return vertex_normals

    @property
    def th_hands_mean(self):
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self):
        return self._mano_layer.th_selected_comps

    @property
    def th_v_template(self):
        return self._mano_layer.th_v_template

    @property
    def side(self):
        return self._side

    @property
    def num_verts(self):
        return 778
