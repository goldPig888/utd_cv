import rospy
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json
import time

from pathlib import Path
from dask import delayed, compute


class RosbagExtractor:
    def __init__(self, base_directory, output_directory, debug=False) -> None:
        rospy.init_node('extract_rosbag', anonymous=True)
        self.base_directory = Path(base_directory)
        self.output_directory = Path(output_directory)
        self.bag_files = list(self.base_directory.glob("**/*.bag"))
        self.bridge = CvBridge()
        self.debug = debug
        self._process_bags_concurrent()

    def _get_camera_topics(self, topics):
        cameras = {}
        for topic in topics:
            if 'image_raw' in topic:
                parts = topic.split('/')
                camera_id = parts[1]
                if camera_id not in cameras:
                    cameras[camera_id] = {'color': None, 'depth': None}
                if 'color' in topic and 'aligned_depth_to_color' not in topic:
                    cameras[camera_id]['color'] = topic
                elif 'aligned_depth_to_color' in topic:
                    cameras[camera_id]['depth'] = topic
        return cameras

    @delayed
    def _save_image(self, msg, filename, is_color=True):
        try:
            if is_color:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img = cv_img.astype(np.uint16)

            _, img_encoded = cv2.imencode('.jpg' if is_color else '.png', cv_img)
            with open(str(filename), 'wb') as f:
                f.write(img_encoded.tobytes())
        except CvBridgeError as e:
            if self.debug:
                print(f"Error converting image: {e}")
        except Exception as e:
            if self.debug:
                print(f"Unexpected error: {e}")

    @delayed
    def _process_camera(self, bag_file, output_dir, color_topic, depth_topic, metadata):
        output_dir.mkdir(parents=True, exist_ok=True)
        num_frames = 0

        with rosbag.Bag(str(bag_file), 'r') as bag:
            tasks = []
            for topic, msg, _ in bag.read_messages(topics=[color_topic, depth_topic]):
                if topic == color_topic:
                    color_filename = output_dir / f"color_{num_frames:06}.jpg"
                    tasks.append(self._save_image(msg, color_filename, is_color=True))
                elif topic == depth_topic:
                    depth_filename = output_dir / f"depth_{num_frames:06}.png"
                    tasks.append(self._save_image(msg, depth_filename, is_color=False))
                num_frames += 1

        metadata['num_frames'] = num_frames
        compute(*tasks)

    @delayed
    def _process_bag(self, bag_file):
        bag_name = bag_file.stem
        person_id = bag_file.parts[-3]

        master_output_dir = self.output_directory / bag_name
        master_output_dir.mkdir(parents=True, exist_ok=True)

        with rosbag.Bag(str(bag_file), 'r') as bag:
            topic_info = bag.get_type_and_topic_info()
            topics = topic_info[1].keys()

        cameras = self._get_camera_topics(topics)

        tasks = []
        for camera_id, topics in cameras.items():
            if topics['color'] and topics['depth']:
                output_dir = master_output_dir / camera_id
                metadata = {
                    "serials": list(cameras.keys()),
                    "width": 640,
                    "height": 480,
                    "extrinsics": "extrinsics_20240611",
                    "mano_calib": person_id,
                    "object_ids": ["test"],
                    "mano_sides": ["right", "left"],
                    "num_frames": 0
                }
                tasks.append(self._process_camera(bag_file, output_dir, topics['color'], topics['depth'], metadata))
                self._generate_metadata(metadata, master_output_dir)

        compute(*tasks)

    def _process_bags_concurrent(self):
        tasks = [self._process_bag(bag_file) for bag_file in self.bag_files]
        compute(*tasks)

    def _generate_metadata(self, metadata, output_dir):
        meta_file = output_dir / "meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        if self.debug:
            print(f"Metadata written to {meta_file}")

def main():
    base_directory = "/Users/mayespinola/Documents/GitHub/utd_cv/data/rosbags"
    output_directory = "/Users/mayespinola/Documents/GitHub/utd_cv/data/recordings"
    RosbagExtractor(base_directory, output_directory, debug=True)
    print("done")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")
