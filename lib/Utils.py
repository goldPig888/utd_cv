from pathlib import Path
import shutil
import json
import math
import sys
import av
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import concurrent.futures
import multiprocessing
import time
from typing import List, Tuple, Dict, Any
from tqdm import tqdm


from .Colors import (
    OBJ_CLASS_COLORS,
    HAND_COLORS,
    HAND_BONE_COLORS,
    HAND_JOINT_COLORS,
    COLORS,
)
from .ManoInfo import *

PROJ_ROOT = Path(__file__).resolve().parents[1]  # Get the project root directory
EXTERNAL_ROOT = PROJ_ROOT / "externals"

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def get_logger(log_name="default", log_level="INFO"):
    import logging

    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = levels.get(log_level.lower(), logging.INFO)

    logger = logging.getLogger(log_name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)-20s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def make_clean_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def read_data_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_data_to_json(json_path, data):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=False)


def apply_transformation(points, trans_mat):
    homo_points = np.stack([points, np.ones((points.shape[0], 1))])
    homo_points = homo_points.dot(trans_mat.T)
    return homo_points[:, :3]


def rvt_to_quat(rvt):
    """Convert rotation vector and translation vector to quaternion and translation vector.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) or (N, 6). [rvx, rvy, rvz, tx, ty, tz]

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) or (N, 7), [qx, qy, qz, qw, tx, ty, tz].

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(rvt, np.ndarray) or rvt.shape[-1] != 6:
        raise ValueError("Input must be a numpy array with last dimension size 6.")

    if rvt.ndim == 2:
        rv = rvt[:, :3]
        t = rvt[:, 3:]
    elif rvt.ndim == 1:
        rv = rvt[:3]
        t = rvt[3:]
    else:
        raise ValueError(
            "Input must be either 1D or 2D with a last dimension size of 6."
        )

    r = R.from_rotvec(rv)
    q = r.as_quat()  # this will be (N, 4) if rv is (N, 3), otherwise (4,)
    if q.ndim == 1:
        return np.concatenate((q, t))  # 1D case
    return np.concatenate((q, t), axis=-1)  # 2D case


def rvt_to_mat(rvt):
    """Convert rotation vector and translation vector to pose matrix.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) or (N, 6). [rvx, rvy, rvz, tx, ty, tz]
    Returns:
        np.ndarray: Pose matrix, shape (4, 4) or (N, 4, 4).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(rvt, np.ndarray) or rvt.shape[-1] != 6:
        raise ValueError("Input must be a numpy array with last dimension size 6.")

    if rvt.ndim == 1:
        p = np.eye(4)
        rv = rvt[:3]
        t = rvt[3:]
        r = R.from_rotvec(rv)
        p[:3, :3] = r.as_matrix()
        p[:3, 3] = t
        return p.astype(np.float32)
    elif rvt.ndim == 2:
        p = np.eye(4).reshape((1, 4, 4)).repeat(len(rvt), axis=0)
        rv = rvt[:, :3]
        t = rvt[:, 3:]
        r = R.from_rotvec(rv)
        for i in range(len(rvt)):
            p[i, :3, :3] = r[i].as_matrix()
            p[i, :3, 3] = t[i]
        return p.astype(np.float32)
    else:
        raise ValueError(
            "Input must be either 1D or 2D with a last dimension size of 6."
        )


def mat_to_rvt(mat_4x4):
    """Convert pose matrix to rotation vector and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) or (N, 4, 4).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) or (N, 6).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """

    if isinstance(mat_4x4, list):
        mat_4x4 = np.array(mat_4x4)

    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if mat_4x4.ndim == 2:  # Single matrix input
        r = R.from_matrix(mat_4x4[:3, :3])
        rv = r.as_rotvec()
        t = mat_4x4[:3, 3]
        return np.concatenate([rv, t], dtype=np.float32)
    elif mat_4x4.ndim == 3:  # Batch of matrices
        rv = np.empty((len(mat_4x4), 3), dtype=np.float32)
        t = mat_4x4[:, :3, 3]
        for i, mat in enumerate(mat_4x4):
            r = R.from_matrix(mat[:3, :3])
            rv[i] = r.as_rotvec()
        return np.concatenate([rv, t], axis=1, dtype=np.float32)
    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")


def mat_to_quat(mat_4x4):
    """Convert pose matrix to quaternion and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) or (N, 4, 4).
    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) or (N, 7).

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if mat_4x4.ndim == 2:  # Single matrix
        r = R.from_matrix(mat_4x4[:3, :3])
        q = r.as_quat()
        t = mat_4x4[:3, 3]
    elif mat_4x4.ndim == 3:  # Batch of matrices
        r = R.from_matrix(mat_4x4[:, :3, :3])
        q = r.as_quat()
        t = mat_4x4[:, :3, 3]
    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")

    return np.concatenate([q, t], axis=-1).astype(np.float32)


def quat_to_mat(quat):
    """Convert quaternion and translation vector to pose matrix.

    Args:
        quat (np.ndarray): Quaternion and translation vector, shape (7,) or (N, 7).
    Returns:
        np.ndarray: Pose matrix, shape (4, 4) or (N, 4, 4).

    Raises:
        ValueError: If the input does not have the last dimension size of 7.
    """
    if quat.shape[-1] != 7:
        raise ValueError("Input must have the last dimension size of 7.")

    batch_mode = quat.ndim == 2
    q = quat[..., :4]
    t = quat[..., 4:]

    if batch_mode:
        p = np.eye(4).reshape(1, 4, 4).repeat(len(quat), axis=0)
    else:
        p = np.eye(4)

    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()
    p[..., :3, 3] = t

    return p.astype(np.float32)


def quat_to_rvt(quat):
    """Convert quaternion and translation vector to rotation vector and translation vector.

    Args:
        quat (np.ndarray): Quaternion and translation vector, shape (7,) or (N, 7).
    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) or (N, 6).

    Raises:
        ValueError: If the input does not have the last dimension size of 7.
    """
    if isinstance(quat, list):
        quat = np.array(quat)

    if quat.shape[-1] != 7:
        raise ValueError("Input must have the last dimension size of 7.")

    batch_mode = quat.ndim == 2
    q = quat[..., :4]
    t = quat[..., 4:]

    r = R.from_quat(q)
    rv = r.as_rotvec()

    if batch_mode:
        return np.concatenate(
            [rv, t], axis=-1, dtype=np.float32
        )  # Ensure that the right axis is used for batch processing
    else:
        return np.concatenate([rv, t], dtype=np.float32)  # No axis needed for 1D arrays


def read_rgb_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_rgb_image(image_path, image):
    cv2.imwrite(str(image_path), image[:, :, ::-1])


def read_depth_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    return image


def write_depth_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def read_mask_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    return image


def write_mask_image(image_path, image):
    cv2.imwrite(str(image_path), image)


def create_video_from_rgb_images(video_path, images, fps=30):
    """
    Create a video from a list of RGB images using H.264 codec with PyAV.

    Args:
        video_path (str): Path to save the output video.
        images (list of ndarray): List of RGB images to include in the video.
        fps (int, optional): Frames per second for the video. Defaults to 30.

    Returns:
        None
    """
    if not images:
        raise ValueError("The images list is empty.")

    # Ensure all images have the same shape
    height, width, _ = images[0].shape
    for image in images:
        if image.shape != (height, width, 3):
            raise ValueError("All images must have the same dimensions and be RGB.")

    # Create a video container
    container = av.open(video_path, mode="w")

    # Create a video stream
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for image in images:
        # Convert image from RGB to YUV420
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame = frame.reformat(format="yuv420p")

        # Encode the frame
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush and close the container
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def erode_mask(mask, kernel_size=3, iterations=1):
    """
    Erode a mask image using a specified kernel size and number of iterations.

    Args:
        mask (np.ndarray): The mask image to be eroded.
        kernel_size (int, optional): Size of the erosion kernel. Default is 3.
        iterations (int, optional): Number of erosion iterations. Default is 1.

    Returns:
        np.ndarray: The eroded mask image.
    """
    # If the kernel size is less than or equal to 1, return the original mask
    if kernel_size <= 1:
        return mask

    # Create the erosion kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Convert the mask to uint8 if necessary
    original_dtype = mask.dtype
    mask_uint8 = mask.astype(np.uint8)

    # Apply the erosion
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)

    # Convert the eroded mask back to its original data type
    return eroded_mask.astype(original_dtype)


def dilate_mask(mask, kernel_size=3, iterations=1):
    """
    Dilate a mask image using a specified kernel size and number of iterations.

    Args:
        mask (np.ndarray): The mask image to be dilated.
        kernel_size (int, optional): Size of the dilation kernel. Default is 3.
        iterations (int, optional): Number of dilation iterations. Default is 1.

    Returns:
        np.ndarray: The dilated mask image.
    """
    # If the kernel size is less than or equal to 1, return the original mask
    if kernel_size <= 1:
        return mask

    # Create the dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Convert the mask to uint8 if necessary
    original_dtype = mask.dtype
    mask_uint8 = mask.astype(np.uint8)

    # Apply the dilation
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)

    # Convert the dilated mask back to its original data type
    return dilated_mask.astype(original_dtype)


def adjust_xyxy_bbox(bbox, width, height, margin=3):
    """
    Adjust bounding box coordinates with margins and boundary conditions.

    Args:
        bbox (list of int or float): Bounding box coordinates [x_min, y_min, x_max, y_max].
        width (int): Width of the image or mask. Must be greater than 0.
        height (int): Height of the image or mask. Must be greater than 0.
        margin (int): Margin to be added to the bounding box. Must be non-negative.

    Returns:
        np.ndarray: Adjusted bounding box as a numpy array.

    Raises:
        ValueError: If inputs are not within the expected ranges or types.
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain exactly four coordinates.")
    if not all(isinstance(x, (int, float)) for x in bbox):
        raise ValueError("Bounding box coordinates must be integers or floats.")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or not isinstance(margin, int)
    ):
        raise ValueError("Width, height, and margin must be integers.")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if margin < 0:
        raise ValueError("Margin must be a non-negative integer.")

    # Convert bbox to integers if necessary
    x_min, y_min, x_max, y_max = map(int, bbox)

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width - 1, x_max + margin)
    y_max = min(height - 1, y_max + margin)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


def get_bbox_from_landmarks(landmarks, width, height, margin=5):
    """
    Calculate a bounding box from a set of landmarks, adding an optional margin.

    Args:
        landmarks (np.ndarray): Landmarks array, shape (num_points, 2), with points marked as [-1, -1] being invalid.
        width (int): Width of the image or frame from which landmarks were extracted.
        height (int): Height of the image or frame.
        margin (int): Margin to add around the calculated bounding box. Default is 3.

    Returns:
        np.ndarray: Bounding box coordinates as [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If landmarks are not a 2D numpy array with two columns, or if width, height, or margin are non-positive.
    """
    if (
        not isinstance(landmarks, np.ndarray)
        or landmarks.ndim != 2
        or landmarks.shape[1] != 2
    ):
        raise ValueError(
            "Landmarks must be a 2D numpy array with shape (num_points, 2)."
        )
    if not all(isinstance(i, int) and i > 0 for i in [width, height, margin]):
        raise ValueError("Width, height, and margin must be positive integers.")

    valid_marks = landmarks[~np.any(landmarks == -1, axis=1)]
    if valid_marks.size == 0:
        return np.array([-1, -1, -1, -1], dtype=np.int64)

    x, y, w, h = cv2.boundingRect(valid_marks)
    bbox = np.array(
        [x - margin, y - margin, x + w + margin, y + h + margin], dtype=np.int64
    )
    bbox[:2] = np.maximum(0, bbox[:2])  # Adjust x_min and y_min
    bbox[2] = min(width - 1, bbox[2])  # Adjust x_max
    bbox[3] = min(height - 1, bbox[3])  # Adjust y_max

    return bbox


def get_bbox_from_mask(mask, margin=3):
    """
    Calculate a bounding box from a binary mask with an optional margin.

    Args:
        mask (np.ndarray): Binary mask, shape (height, width), where non-zero values indicate areas of interest.
        margin (int): Margin to add around the calculated bounding box. Must be non-negative.

    Returns:
        np.ndarray: Adjusted bounding box coordinates as [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If the mask is not a 2D array or contains no non-zero values, or if the margin is negative.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D numpy array.")
    if margin < 0:
        raise ValueError("Margin must be non-negative.")
    if not np.any(mask):
        raise ValueError(
            "Mask contains no non-zero values; cannot determine bounding rectangle."
        )

    height, width = mask.shape
    mask_uint8 = mask.astype(
        np.uint8
    )  # Ensure mask is in appropriate format for cv2.boundingRect
    x, y, w, h = cv2.boundingRect(mask_uint8)
    bbox = [x, y, x + w, y + h]
    bbox[0] = max(0, bbox[0] - margin)
    bbox[1] = max(0, bbox[1] - margin)
    bbox[2] = min(width - 1, bbox[2] + margin)
    bbox[3] = min(height - 1, bbox[3] + margin)

    return np.array(bbox, dtype=np.int64)


def xyxy_to_cxcywh(bbox):
    """
    Convert bounding box coordinates from top-left and bottom-right (XYXY) format
    to center-x, center-y, width, and height (CXCYWH) format.

    Args:
        bbox (np.ndarray): Bounding box array in XYXY format. Should be of shape (4,) for a single box
                            or (N, 4) for multiple boxes, where N is the number of boxes.

    Returns:
        np.ndarray: Converted bounding box array in CXCYWH format, with the same shape as the input.

    Raises:
        ValueError: If the input is not 1D or 2D with the last dimension size of 4.
    """
    bbox = np.asarray(bbox)
    if bbox.ndim not in [1, 2] or bbox.shape[-1] != 4:
        raise ValueError(
            "Input array must be 1D or 2D with the last dimension size of 4."
        )

    # Calculate the center coordinates, width, and height
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]

    return np.stack([cx, cy, w, h], axis=-1)


def display_images(
    images,
    names=None,
    cmap="gray",
    figsize=(19.2, 10.8),
    dpi=100,
    max_cols=4,
    facecolor="white",
    save_path=None,
    return_array=False,
):
    img_array = None
    num_images = len(images)
    num_cols = min(num_images, max_cols)
    num_rows = (
        num_images + num_cols - 1
    ) // num_cols  # More efficient ceiling division
    if names is None:
        names = [f"fig_{i}" for i in range(num_images)]

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=figsize, dpi=dpi, facecolor=facecolor
    )

    # Flatten axs for easy iteration and handle cases with single subplot
    if num_images == 1:
        axs = [axs]
    else:
        axs = axs.flat

    try:
        for i, (image, name) in enumerate(zip(images, names)):
            ax = axs[i]
            if image.ndim == 3 and image.shape[2] == 3:  # RGB images
                ax.imshow(image)
            else:  # Depth or grayscale images
                ax.imshow(image, cmap=cmap)
            ax.set_title(name)
            ax.axis("off")

        # Hide any unused axes
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        if return_array:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img_array = np.array(canvas.renderer.buffer_rgba())[:, :, :3]

        if save_path:
            plt.savefig(str(save_path))

        if not save_path and not return_array:
            plt.show()

    finally:
        plt.close(fig)

    if return_array:
        return img_array


def draw_hand_landmarks(image, landmarks, hand_side=None, box=None):
    """
    Draws hand landmarks, bones, and optional hand side text and bounding box on an image.

    Args:
        image (np.ndarray): The image on which to draw the landmarks.
        landmarks (np.ndarray): Array of hand landmarks.
        hand_side (str, optional): Indicates 'left' or 'right' hand. Default is None.
        box (tuple, optional): Bounding box coordinates as (x1, y1, x2, y2). Default is None.

    Returns:
        np.ndarray: Image with drawn hand landmarks.
    """
    img = image.copy()

    # Draw bones
    for idx, bone in enumerate(HAND_BONES):
        start, end = landmarks[bone[0]], landmarks[bone[1]]
        if np.any(start == -1) or np.any(end == -1):
            continue
        color = HAND_BONE_COLORS[idx].rgb
        cv2.line(img, tuple(start), tuple(end), color, 2)

    # Draw joints
    for idx, mark in enumerate(landmarks):
        if np.any(mark == -1):
            continue
        cv2.circle(img, tuple(mark), 5, (255, 255, 255), -1)  # White base for joints
        color = HAND_JOINT_COLORS[idx].rgb
        cv2.circle(img, tuple(mark), 3, color, -1)

    # Draw bounding box
    if box:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Draw hand side text
    if hand_side:
        text = hand_side.lower()
        text_x = np.min(landmarks[:, 0])
        text_y = np.min(landmarks[:, 1]) - 5  # Add margin to top
        text_color = HAND_COLORS[1] if text == "right" else HAND_COLORS[2]
        cv2.putText(
            img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            text_color.rgb,
            1,
            cv2.LINE_AA,
        )

    return img


def draw_losses_curve(
    loss_lists,
    labels=None,
    title="Loss Curves",
    xlabel="Epoch",
    ylabel="Loss",
    figsize=(19.2, 10.8),
    dpi=100,
    save_path=None,
):
    """
    Plot multiple loss curves.

    Args:
        loss_lists (list of lists): List of lists, where each inner list contains loss values for different metrics.
        labels (list of str, optional): List of labels for each loss curve. Default is None.
        title (str): Title of the plot. Default is "Loss Curves".
        xlabel (str): Label for the x-axis. Default is "Epoch".
        ylabel (str): Label for the y-axis. Default is "Loss".
        figsize (tuple, optional): Size of the figure. Default is (19.2, 10.8).
        dpi (int, optional): Dots per inch for the figure. Default is 100.
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed. Default is None.

    Returns:
        None
    """
    # Create a new figure with the specified size and DPI
    plt.figure(figsize=figsize, dpi=dpi)

    # Plot each loss curve with an appropriate label
    for i, loss_list in enumerate(loss_lists):
        label = labels[i] if labels and i < len(labels) else f"Loss {i+1}"
        plt.plot(loss_list, label=label)

    # Set the title and labels for the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Display the legend and grid
    plt.legend()
    plt.grid(True)

    # Save the plot to a file if save_path is provided, otherwise display it
    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

    # Close the figure to free memory
    plt.close()


def draw_mask_overlay(rgb, mask, alpha=0.5, mask_color=(0, 255, 0), reduce_bg=False):
    """
    Draw a mask overlay on an RGB image.

    Args:
        rgb (np.ndarray): RGB image, shape (height, width, 3).
        mask (np.ndarray): Binary mask, shape (height, width).
        alpha (float): Transparency of the mask overlay.
        mask_color (tuple): RGB color of the mask overlay.
        reduce_bg (bool): If True, reduce the background intensity of the RGB image.

    Returns:
        np.ndarray: RGB image with mask overlay.
    """
    # Create an overlay based on whether to reduce the background
    if reduce_bg:
        overlay = np.zeros_like(rgb)
    else:
        overlay = rgb.copy()

    # Apply the mask color to the overlay where the mask is true
    overlay[mask.astype(bool)] = mask_color

    # Blend the overlay with the original image
    blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)

    return blended


# def draw_debug_image(
#     rgb_image,
#     hand_mask=None,
#     object_mask=None,
#     prompt_points=None,
#     prompt_labels=None,
#     hand_marks=None,
#     alpha=0.5,
#     reduce_background=False,
#     draw_boxes=False,
#     draw_hand_sides=False,
# ):
#     height, width = rgb_image.shape[:2]
#     overlay = np.zeros_like(rgb_image) if reduce_background else rgb_image.copy()

#     # draw hand mask
#     if hand_mask is not None:
#         for label in np.unique(hand_mask):
#             if label == 0:
#                 continue
#             overlay[hand_mask == label] = HAND_COLORS[label].rgb

#     # draw object mask
#     if object_mask is not None:
#         for label in np.unique(object_mask):
#             if label == 0:
#                 continue
#             overlay[object_mask == label] = OBJ_CLASS_COLORS[label].rgb

#     # draw boxes
#     if draw_boxes:
#         if hand_mask is not None:
#             for label in np.unique(hand_mask):
#                 if label == 0:
#                     continue
#                 mask = hand_mask == label
#                 color = HAND_COLORS[label]
#                 box = get_bbox_from_mask(mask)
#                 cv2.rectangle(
#                     overlay,
#                     (box[0], box[1]),
#                     (box[2], box[3]),
#                     color.rgb,
#                     2,
#                 )
#         if object_mask is not None:
#             for label in np.unique(object_mask):
#                 if label == 0:
#                     continue
#                 mask = object_mask == label
#                 box = get_bbox_from_mask(mask)
#                 color = OBJ_CLASS_COLORS[label]
#                 cv2.rectangle(
#                     overlay,
#                     (box[0], box[1]),
#                     (box[2], box[3]),
#                     color.rgb,
#                     2,
#                 )

#     # draw prompt points
#     if prompt_points is not None and prompt_labels is not None:
#         points = np.array(prompt_points, dtype=np.int32).reshape(-1, 2)
#         labels = np.array(prompt_labels, dtype=np.int32).reshape(-1)
#         for i, (point, label) in enumerate(zip(points, labels)):
#             color = COLORS["dark_red"] if label == 0 else COLORS["dark_green"]
#             cv2.circle(overlay, point, 3, color.rgb, -1)

#     overlay = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)

#     # draw hand sides for hand mask
#     if draw_hand_sides and hand_mask is not None and hand_marks is None:
#         for label in np.unique(hand_mask):
#             if label == 0:
#                 continue
#             mask = hand_mask == label
#             color = HAND_COLORS[label]
#             text = "right" if label == 1 else "left"
#             x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
#             text_x = x
#             text_y = y - 5
#             cv2.putText(
#                 overlay,
#                 text,
#                 (text_x, text_y),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 1,
#                 color.rgb,
#                 1,
#                 cv2.LINE_AA,
#             )

#     # draw hand landmarks
#     if hand_marks is not None:
#         for ind, marks in enumerate(hand_marks):
#             if np.all(marks == -1):
#                 continue

#             # draw bones
#             for bone_idx, bone in enumerate(HAND_BONES):
#                 if np.any(marks[bone[0]] == -1) or np.any(marks[bone[1]] == -1):
#                     continue
#                 color = HAND_BONE_COLORS[bone_idx]
#                 cv2.line(
#                     overlay,
#                     marks[bone[0]],
#                     marks[bone[1]],
#                     color.rgb,
#                     2,
#                 )
#             # draw joints
#             for i, mark in enumerate(marks):
#                 if np.any(mark == -1):
#                     continue
#                 color = HAND_JOINT_COLORS[i]
#                 cv2.circle(overlay, mark, 5, (255, 255, 255), -1)
#                 cv2.circle(
#                     overlay,
#                     mark,
#                     3,
#                     color.rgb,
#                     -1,
#                 )

#             if draw_boxes:
#                 box = get_bbox_from_landmarks(marks, width, height)
#                 color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
#                 cv2.rectangle(
#                     overlay,
#                     (box[0], box[1]),
#                     (box[2], box[3]),
#                     color.rgb,
#                     2,
#                 )

#             if draw_hand_sides:
#                 text = "right" if ind == 0 else "left"
#                 color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
#                 x, y, _, _ = cv2.boundingRect(
#                     np.array([m for m in marks if np.all(m != -1)], dtype=np.int64)
#                 )
#                 text_x = x
#                 text_y = y - 5
#                 cv2.putText(
#                     overlay,
#                     text,
#                     (text_x, text_y),
#                     cv2.FONT_HERSHEY_DUPLEX,
#                     1,
#                     color.rgb,
#                     1,
#                     cv2.LINE_AA,
#                 )

#     return overlay


def draw_debug_image(
    rgb_image,
    hand_mask=None,
    object_mask=None,
    prompt_points=None,
    prompt_labels=None,
    hand_marks=None,
    alpha=0.5,
    reduce_background=False,
    draw_boxes=False,
    draw_hand_sides=False,
):
    """
    Draws debug information on an RGB image.

    Args:
        rgb_image (np.ndarray): The original RGB image.
        hand_mask (np.ndarray, optional): Mask of the hands.
        object_mask (np.ndarray, optional): Mask of the objects.
        prompt_points (list, optional): Points to be drawn on the image.
        prompt_labels (list, optional): Labels for the prompt points.
        hand_marks (list, optional): Hand landmark points.
        alpha (float, optional): Transparency factor for overlay. Defaults to 0.5.
        reduce_background (bool, optional): Whether to reduce the background visibility. Defaults to False.
        draw_boxes (bool, optional): Whether to draw bounding boxes around hands and objects. Defaults to False.
        draw_hand_sides (bool, optional): Whether to draw text indicating left/right hand. Defaults to False.

    Returns:
        np.ndarray: The image with debug information drawn on it.
    """
    height, width = rgb_image.shape[:2]
    overlay = np.zeros_like(rgb_image) if reduce_background else rgb_image.copy()

    def apply_mask(mask, colors):
        for label in np.unique(mask):
            if label == 0:
                continue
            overlay[mask == label] = colors[label].rgb

    def draw_boxes_from_mask(mask, colors):
        for label in np.unique(mask):
            if label == 0:
                continue
            box = get_bbox_from_mask(mask == label)
            cv2.rectangle(
                overlay, (box[0], box[1]), (box[2], box[3]), colors[label].rgb, 2
            )

    # Draw hand mask
    if hand_mask is not None:
        apply_mask(hand_mask, HAND_COLORS)

    # Draw object mask
    if object_mask is not None:
        apply_mask(object_mask, OBJ_CLASS_COLORS)

    # Draw bounding boxes
    if draw_boxes:
        if hand_mask is not None:
            draw_boxes_from_mask(hand_mask, HAND_COLORS)
        if object_mask is not None:
            draw_boxes_from_mask(object_mask, OBJ_CLASS_COLORS)

    # Draw prompt points
    if prompt_points is not None and prompt_labels is not None:
        points = np.array(prompt_points, dtype=np.int32).reshape(-1, 2)
        labels = np.array(prompt_labels, dtype=np.int32).reshape(-1)
        for point, label in zip(points, labels):
            color = COLORS["dark_red"] if label == 0 else COLORS["dark_green"]
            cv2.circle(overlay, tuple(point), 3, color.rgb, -1)

    overlay = cv2.addWeighted(rgb_image, 1 - alpha, overlay, alpha, 0)

    # Draw hand sides
    if draw_hand_sides and hand_mask is not None and hand_marks is None:
        for label in np.unique(hand_mask):
            if label == 0:
                continue
            mask = hand_mask == label
            color = HAND_COLORS[label]
            text = "right" if label == 1 else "left"
            x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
            cv2.putText(
                overlay,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                color.rgb,
                1,
                cv2.LINE_AA,
            )

    # Draw hand landmarks
    if hand_marks is not None:
        for ind, marks in enumerate(hand_marks):
            if np.all(marks == -1):
                continue

            # Draw bones
            for bone_idx, (start, end) in enumerate(HAND_BONES):
                if np.any(marks[start] == -1) or np.any(marks[end] == -1):
                    continue
                color = HAND_BONE_COLORS[bone_idx]
                cv2.line(overlay, tuple(marks[start]), tuple(marks[end]), color.rgb, 2)

            # Draw joints
            for i, mark in enumerate(marks):
                if np.any(mark == -1):
                    continue
                color = HAND_JOINT_COLORS[i]
                cv2.circle(overlay, tuple(mark), 5, (255, 255, 255), -1)
                cv2.circle(overlay, tuple(mark), 3, color.rgb, -1)

            if draw_boxes:
                box = get_bbox_from_landmarks(marks, width, height, margin=10)
                color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color.rgb, 2)

            if draw_hand_sides:
                text = "right" if ind == 0 else "left"
                color = HAND_COLORS[1] if ind == 0 else HAND_COLORS[2]
                x, y, _, _ = cv2.boundingRect(
                    np.array([m for m in marks if np.all(m != -1)], dtype=np.int64)
                )
                cv2.putText(
                    overlay,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    color.rgb,
                    1,
                    cv2.LINE_AA,
                )

    return overlay


class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = torch.zeros(dim_x, dtype=torch.float32)
        self.P = torch.eye(dim_x, dtype=torch.float32) * 10.0
        self.Q = torch.eye(dim_x, dtype=torch.float32) * 0.1
        self.R = torch.eye(dim_z, dtype=torch.float32) * 0.1
        self.F = torch.eye(dim_x, dtype=torch.float32)
        self.H = torch.eye(dim_z, dtype=torch.float32)

    def predict(self):
        self.x = torch.matmul(self.F, self.x)
        self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        z = torch.tensor(z, dtype=torch.float32)
        y = z - torch.matmul(self.H, self.x)
        S = torch.matmul(self.H, torch.matmul(self.P, self.H.T)) + self.R
        K = torch.matmul(torch.matmul(self.P, self.H.T), torch.inverse(S))
        self.x = self.x + torch.matmul(K, y)
        I = torch.eye(self.dim_x, dtype=torch.float32)
        self.P = torch.matmul((I - torch.matmul(K, self.H)), self.P)


def interpolate_missing_poses_kalman(poses):
    poses = torch.tensor(poses, dtype=torch.float32)
    N, D = poses.shape
    assert D == 6, "Each pose should have 6 values: (rx, ry, rz, x, y, z)"

    kf = KalmanFilter(dim_x=D, dim_z=D)

    valid_indices = torch.where(~torch.all(poses == -1, dim=1))[0]
    first_valid_index = valid_indices[0]
    kf.x = poses[first_valid_index]

    interpolated_poses = []

    for i in range(N):
        if torch.all(poses[i] == -1):
            kf.predict()
        else:
            kf.update(poses[i])

        interpolated_poses.append(kf.x.clone().detach().numpy())

    return np.array(interpolated_poses)


def cubic_spline_interpolation(poses):
    import torch.nn.functional as F

    poses = torch.tensor(poses, dtype=torch.float32)
    N, D = poses.shape
    assert D == 6, "Each pose should have 6 values: (rx, ry, rz, x, y, z)"

    valid_indices = torch.where(~torch.all(poses == -1, dim=1))[0]
    invalid_indices = torch.where(torch.all(poses == -1, dim=1))[0]

    for i in range(D):
        valid_values = poses[valid_indices, i]
        cubic_interp = torch.interp1d(valid_indices.float(), valid_values, kind="cubic")
        poses[invalid_indices, i] = cubic_interp(invalid_indices.float())

    return poses.numpy()


def linear_interpolation(poses):
    poses = torch.tensor(poses, dtype=torch.float32)
    N, D = poses.shape
    assert D == 6, "Each pose should have 6 values: (rx, ry, rz, x, y, z)"

    valid_indices = torch.where(~torch.all(poses == -1, dim=1))[0]
    invalid_indices = torch.where(torch.all(poses == -1, dim=1))[0]

    for i in range(D):
        valid_values = poses[valid_indices, i]
        interp_func = torch.interp1d(valid_indices.float(), valid_values)
        poses[invalid_indices, i] = interp_func(invalid_indices.float())

    return poses.numpy()
