# Rospy imports
import rosbag
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
import numpy as np
import json

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

class RosbagExtractor:
    def __init__(self, base_directory, debug=False) -> None:
        rospy.init_node('extract_rosbag', anonymous=True)
        self.base_directory = base_directory
        self.bag_files = self._get_files()
        self.output_directory = "/home/jikaiwang/GitHub/summer_camp/data/recordings"
        self.bridge = CvBridge()
        self._process_bags_concurrent()

    def _get_files(self):
        return sorted([f for f in Path(self.base_directory).glob("**/*.bag")])

    def _read_bag_topic_list(self, bag_file):
        bag_path = str(bag_file)
        with rosbag.Bag(bag_path, 'r') as bag:
            topic_list = set(topic for topic, _, _ in bag.read_messages())
        return list(topic_list)

    def _get_camera_topics(self, topics):
        cameras = {}
        for topic in topics:
            parts = topic.split('/')
            if len(parts) > 2 and 'image_raw' in parts[-1]:
                camera_id = parts[1]
                if camera_id not in cameras:
                    cameras[camera_id] = {}
                if '/color/image_raw' in topic and 'aligned_depth_to_color' not in topic:
                    cameras[camera_id]['color'] = topic
                elif '/aligned_depth_to_color/image_raw' in topic:
                    cameras[camera_id]['depth'] = topic
        return cameras

    def _save_image(self, msg, filename, is_color=True):
        try:
            if is_color:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img = cv_img.astype(np.uint16)

            cv2.imwrite(filename, cv_img)
        except CvBridgeError as e:
            print(f"Error converting image: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def _process_camera(self, bag_file, output_dir, color_topic, depth_topic, metadata):
        bag_path = str(bag_file)

        os.makedirs(output_dir, exist_ok=True)

        color_msgs = []
        depth_msgs = []
        num_frames = 0

        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, _ in bag.read_messages(topics=[color_topic, depth_topic]):
                if topic == color_topic:
                    color_msgs.append(msg)
                elif topic == depth_topic:
                    depth_msgs.append(msg)

        for color_msg, depth_msg in zip(color_msgs, depth_msgs):
            color_filename = os.path.join(output_dir, f"color_{num_frames:06}.jpg")
            depth_filename = os.path.join(output_dir, f"depth_{num_frames:06}.png")
            self._save_image(color_msg, color_filename, is_color=True)
            self._save_image(depth_msg, depth_filename, is_color=False)
            num_frames += 1

        metadata['num_frames'] = num_frames

    def _process_bag(self, bag_file):
        print(f"Processing {bag_file}")
        bag_name = bag_file.stem
        person_id = bag_file.parent.parent.name

        master_output_dir = os.path.join(self.output_directory, bag_name)

        if not os.path.exists(master_output_dir):
            os.makedirs(master_output_dir)

        topics = self._read_bag_topic_list(bag_file)
        cameras = self._get_camera_topics(topics)

        for camera_id, topics in cameras.items():
            if 'color' in topics and 'depth' in topics:
                output_dir = os.path.join(master_output_dir, camera_id)
                print(f"Processing camera: {camera_id}")

                metadata = {
                    "serials": list(cameras.keys()),
                    "width": 640,
                    "height": 480,
                    "extrinsics": "extrinsics_20240611",
                    "mano_calib": person_id,
                    "object_ids": ["G01_1", "G01_2", "G01_3"],
                    "mano_sides": ["right", "left"],
                    "num_frames": 0
                }

                self._process_camera(bag_file, output_dir, topics['color'], topics['depth'], metadata)
                self._generate_metadata(metadata, master_output_dir)
            else:
                print(f"Camera {camera_id} does not have both color and depth topics.")

    def _process_bags_concurrent(self):
        max_workers = os.cpu_count() * 2
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_bag, bag_file) for bag_file in self.bag_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing bag file: {e}")

    def _generate_metadata(self, metadata, output_dir):
        meta_file = os.path.join(output_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata written to {meta_file}")

def main():
    base_directory = "/home/jikaiwang/GitHub/summer_camp/data/rosbags"
    RosbagExtractor(base_directory)
    print("done")

if __name__ == '__main__':
    main()
