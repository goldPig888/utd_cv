import torch
from torch.utils.data import DataLoader
from my_helper import *

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
import os


hamer_checkpoint = f"{PROJ_ROOT}/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
rescale_factor = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model from the checkpoint
hamer_model, hamer_cfg = load_hamer(hamer_checkpoint)

# Move the model to the GPU
hamer_model = hamer_model.to(device)

# Set the model to evaluation mode
hamer_model = hamer_model.eval()

# sequence_folder = PROJ_ROOT / "/home/summer_camp/Desktop/summer_camp/data/recordings/nicole_20240617_102128"
# hand_detection_folder = sequence_folder / "processed/hand_detection"

# meta_file = sequence_folder / "meta.json"
# data = read_data_from_json(meta_file)

# serials = data["serials"]
# width = data["width"]
# height = data["height"]
# num_frames = data["num_frames"]
# mano_sides = data["mano_sides"]

# print(f"Serials: {serials}")
# print(f"Width: {width}")
# print(f"Height: {height}")
# print(f"Num frames: {num_frames}")
# print(f"Mano sides: {mano_sides}")

# landmarks = np.load(hand_detection_folder / "hand_joints_3d_projection.npz")

# # print("landmarks_shape:", np.shape(landmarks))
# # # for item in landmarks.item():
# # #     print(f"item: {item}")
# # cameria_id = ['037522251142', '043422252387', '046122250168', '105322251225', '105322251564', '108222250342', '115422250549', '117222250549']
# # #marks = landmarks["105322251564"][:,100,:]
# # landmarks = np.transpose(landmarks, (2, 1 ,0 ,3 ,4))
# # print("landmarks_shape:", np.shape(landmarks))

# # marks = landmarks[6,100,:]
# # print(f"marks: {marks.shape}, {marks.dtype}")

# # print(marks)



# for key, marks in landmarks.items():
#     print(f"key: {key}, boxes: {marks.shape}, {marks.dtype}")


# marks = landmarks["105322251564"][:,100,:]
# print(f"marks: {marks.shape}, {marks.dtype}")
# print(marks)


# color_file = sequence_folder / "105322251564" / "color_000100.jpg"
# rgb_image = read_rgb_image(color_file)



# # exit()
# # issue: not showing  both hands ( left hand matrix is empty, not extracted from rosbags properly )
# # color_file = sequence_folder / cameria_id[4] / "color_000100.jpg"
# # rgb_image = read_rgb_image(color_file)
# vis_input = draw_debug_image(
#     rgb_image=rgb_image,
#     hand_marks=marks,
#     draw_boxes=True,
#     draw_hand_sides=True,
# )

# display_images([vis_input], ["input marks"])



# input_boxes = []
# right_flags = []


# for i, mks in enumerate(marks):
#     box = get_bbox_from_landmarks(mks, width, height, margin=15)
#     if np.any(box==-1):
#         continue
#     input_boxes.append(box)
#     is_right = i==0
#     right_flags.append(is_right)

# input_boxes = np.array(input_boxes)
# right_flags = np.array(right_flags)

# print(f"input_boxes: {input_boxes.shape}, {input_boxes.dtype}")
# print(input_boxes)

# print(f"right_flags: {right_flags.shape}, {right_flags.dtype}")
# print(right_flags)





# # Create a dataset
# dataset = ViTDetDataset(
#     cfg=hamer_cfg,
#     img_cv2=cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
#     boxes=input_boxes,
#     right=right_flags,
#     rescale_factor=rescale_factor,
# )

# # Create a dataloader for the dataset
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# # Iterate over the dataloader
# hamer_marks = np.full((2, 21, 2), -1, dtype=np.int64)
# with torch.no_grad():
#     for batch in dataloader:
#         # Move the batch to the GPU
#         batch = recursive_to(batch, device)

#         # Forward pass
#         out = hamer_model(batch)

#         # Get the predicted landmarks
#         box_center = batch["box_center"].float()
#         box_size = batch["box_size"].float()
#         right = batch["right"].long()
#         kpts_2d = out["pred_keypoints_2d"].squeeze(0).float()

#         print(f"box_center: {box_center.shape}, {box_center.dtype}, {box_center}")
#         print(f"box_size: {box_size.shape}, {box_size.dtype}, {box_size}")
#         print(f"right: {right.shape}, {right.dtype}, {right}")
#         print(f"kpts_2d: {kpts_2d.shape}, {kpts_2d.dtype}, {kpts_2d}")


#         if right == 0:  # Flip the keypoints if the hand is left
#             kpts_2d[:, 0] *= -1
        
#         # Unnormalize the keypoints
#         kpts_2d = kpts_2d * box_size + box_center
#         kpts_2d = kpts_2d.cpu().numpy()

#         # if right:
#         #     hamer_marks[0] = kpts_2d
#         # else:
#         #     hamer_marks[1] = kpts_2d
#         hamer_marks[0 if right else 1] = kpts_2d




#         vis_hamer = draw_debug_image(
#     rgb_image=rgb_image,
#     hand_marks=hamer_marks,
#     draw_boxes=True,
#     draw_hand_sides=True,
# )

# display_images([vis_input, vis_hamer], ["Input landmarks", "Hamer landmarks"])

# for camera in sequence_folder():
#     color_file = sequence_folder / cameria_id[6] / "color_000100.jpg"
#     print(color_file)


sequence_folder = PROJ_ROOT / "/home/summer_camp/Desktop/summer_camp/data/recordings/may_20240617_101936"
hand_detection_folder = sequence_folder / "processed/hand_detection"

meta_file = sequence_folder / "meta.json"
data = read_data_from_json(meta_file)

serials = data["serials"]
width = data["width"]
height = data["height"]
num_frames = data["num_frames"]
mano_sides = data["mano_sides"]
landmarks = np.load(hand_detection_folder / "hand_joints_3d_projection.npz")

for cam_idx, serial in enumerate(serials):

    for frame_id in range(num_frames):

        marks = landmarks[serial][:,frame_id,:]

        color_file = sequence_folder / serial / f"color_{frame_id:06d}.jpg"
        rgb_image = read_rgb_image(color_file)

        vis_input = draw_debug_image(
            rgb_image=rgb_image,
            hand_marks=marks,
            draw_boxes=True,
            draw_hand_sides=True,
        )


        ######

        #display_images([vis_input], ["input marks"])


        input_boxes = []
        right_flags = []


        for i, mks in enumerate(marks):
            box = get_bbox_from_landmarks(mks, width, height, margin=15)
            if np.any(box==-1):
                continue
            input_boxes.append(box)
            is_right = i==0
            right_flags.append(is_right)

        input_boxes = np.array(input_boxes)
        right_flags = np.array(right_flags)

        # print(f"input_boxes: {input_boxes.shape}, {input_boxes.dtype}")
        # print(input_boxes)

        # print(f"right_flags: {right_flags.shape}, {right_flags.dtype}")
        # print(right_flags)


        # Create a dataset
        dataset = ViTDetDataset(
            cfg=hamer_cfg,
            img_cv2=cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
            boxes=input_boxes,
            right=right_flags,
            rescale_factor=rescale_factor,
        )

        # Create a dataloader for the dataset
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Iterate over the dataloader
        hamer_marks = np.full((2, 21, 2), -1, dtype=np.int64)
        with torch.no_grad():
            for batch in dataloader:
                # Move the batch to the GPU
                batch = recursive_to(batch, device)

                # Forward pass
                out = hamer_model(batch)

                # Get the predicted landmarks
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                right = batch["right"].long()
                kpts_2d = out["pred_keypoints_2d"].squeeze(0).float()

                # print(f"box_center: {box_center.shape}, {box_center.dtype}, {box_center}")
                # print(f"box_size: {box_size.shape}, {box_size.dtype}, {box_size}")
                # print(f"right: {right.shape}, {right.dtype}, {right}")
                # print(f"kpts_2d: {kpts_2d.shape}, {kpts_2d.dtype}, {kpts_2d}")


                if right == 0:  # Flip the keypoints if the hand is left
                    kpts_2d[:, 0] *= -1
                
                # Unnormalize the keypoints
                kpts_2d = kpts_2d * box_size + box_center
                kpts_2d = kpts_2d.cpu().numpy()

                # if right:
                #     hamer_marks[0] = kpts_2d
                # else:
                #     hamer_marks[1] = kpts_2d
                hamer_marks[0 if right else 1] = kpts_2d




            vis_hamer = draw_debug_image(
            rgb_image=rgb_image,
            hand_marks=hamer_marks,
            draw_boxes=True,
            draw_hand_sides=True,
        )

        display_images([vis_input, vis_hamer], ["Input landmarks", "Hamer landmarks"],serial + str(frame_id))

        # for camera in sequence_folder():
        #     color_file = sequence_folder / serial / "color_000100.jpg"
        #     print(color_file)



# directory = '/path/to/your/images/'




# create_video_from_rgb_images("/home/summer_camp/Desktop/summer_camp/hamer", /test_save_images)
