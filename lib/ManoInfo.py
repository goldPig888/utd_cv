# MediaPipe Hands connections
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

# Combined connections for all hand bones
HAND_BONES = (
    HAND_PALM_CONNECTIONS
    + HAND_THUMB_CONNECTIONS
    + HAND_INDEX_FINGER_CONNECTIONS
    + HAND_MIDDLE_FINGER_CONNECTIONS
    + HAND_RING_FINGER_CONNECTIONS
    + HAND_PINKY_FINGER_CONNECTIONS
)

# Names of hand joints
HAND_JOINT_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# Parent joint indices for each hand joint
HAND_JOINT_PARENTS = (
    -1,  # WRIST has no parent
    0,  # THUMB_CMC is connected to WRIST
    1,  # THUMB_MCP is connected to THUMB_CMC
    2,  # THUMB_IP is connected to THUMB_MCP
    3,  # THUMB_TIP is connected to THUMB_IP
    0,  # INDEX_MCP is connected to WRIST
    5,  # INDEX_PIP is connected to INDEX_MCP
    6,  # INDEX_DIP is connected to INDEX_PIP
    7,  # INDEX_TIP is connected to INDEX_DIP
    0,  # MIDDLE_MCP is connected to WRIST
    9,  # MIDDLE_PIP is connected to MIDDLE_MCP
    10,  # MIDDLE_DIP is connected to MIDDLE_PIP
    11,  # MIDDLE_TIP is connected to MIDDLE_DIP
    0,  # RING_MCP is connected to WRIST
    13,  # RING_PIP is connected to RING_MCP
    14,  # RING_DIP is connected to RING_PIP
    15,  # RING_TIP is connected to RING_DIP
    0,  # PINKY_MCP is connected to WRIST
    17,  # PINKY_PIP is connected to PINKY_MCP
    18,  # PINKY_DIP is connected to PINKY_PIP
    19,  # PINKY_TIP is connected to PINKY_DIP
)

# Faces that make the hand mesh watertight
NEW_MANO_FACES_RIGHT = (
    (92, 38, 234),
    (234, 38, 239),
    (38, 122, 239),
    (239, 122, 279),
    (122, 118, 279),
    (279, 118, 215),
    (118, 117, 215),
    (215, 117, 214),
    (117, 119, 214),
    (214, 119, 121),
    (119, 120, 121),
    (121, 120, 78),
    (120, 108, 78),
    (78, 108, 79),
)
NEW_MANO_FACES_LEFT = tuple(face[::-1] for face in NEW_MANO_FACES_RIGHT)

# Constants for MANO vertices and faces
NUM_MANO_VERTS = 778
NUM_MANO_FACES = 1538
