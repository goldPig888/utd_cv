# Rospy imports
import rosbag
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
import numpy as np
import json

class RosbagExtractor:
    def __init__(self, directory, name, debug=False) -> None:
        rospy.init_node('extract_rosbag', anonymous=True)
        self.bag_files = self._get_files(directory)
        self.directory = directory
        self.name = name
        self._process_bags()


    def _get_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.bag')]


    def _read_bag_topic_list(self, bag_file):

        bag_path = os.path.join(self.directory, bag_file)
        bag = rosbag.Bag(bag_path, 'r')

        topic_list = set()
        for topic, _, _ in bag.read_messages():
            topic_list.add(topic)
        bag.close()
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


    def _save_image(self, bridge, msg, filename, is_color=True):
        try:
            if is_color:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(filename, cv_img)
            else:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_array = np.array(cv_img, dtype=np.float32)
                depth_array = depth_array.astype(np.uint16)
                cv2.imwrite(filename, depth_array)
            print("Wrote file: {}".format(filename))
        except CvBridgeError as e:
            print(e)
            print("Error converting image")
        except Exception as e:
            print(f"Unexpected error: {e}")


    def _process_camera(self, bag_file, output_dir, color_topic, depth_topic, metadata):
        bag_path = os.path.join(self.directory, bag_file)
        bag = rosbag.Bag(bag_path, 'r')
        bridge = CvBridge()

        os.makedirs(output_dir, exist_ok=True)

        color_filter = message_filters.SimpleFilter()
        depth_filter = message_filters.SimpleFilter()
        ts = message_filters.ApproximateTimeSynchronizer([color_filter, depth_filter], 10, 0.1, allow_headerless=True)

        color_count = 0
        depth_count = 0
        num_frames = 0

        def callback(color_msg, depth_msg):
            nonlocal color_count, depth_count, num_frames
            color_filename = os.path.join(output_dir, "color_%06i.jpg" % color_count)
            depth_filename = os.path.join(output_dir, "depth_%06i.png" % depth_count)
            self._save_image(bridge, color_msg, color_filename, is_color=True)
            self._save_image(bridge, depth_msg, depth_filename, is_color=False)
            color_count += 1
            depth_count += 1
            num_frames += 1

        ts.registerCallback(callback)

        for topic, msg, _ in bag.read_messages(topics=[color_topic, depth_topic]):
            if topic == color_topic:
                color_filter.signalMessage(msg)
            elif topic == depth_topic:
                depth_filter.signalMessage(msg)

        bag.close()
        metadata['num_frames'] = num_frames


    def _process_bags(self):
        for bag_file in self.bag_files:
            print(bag_file)
            parts = bag_file.split('.')
            master_output_dir = os.path.join("/home/jikaiwang/GitHub/summer_camp/data/recordingsNew", parts[0])

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
                        "mano_calib": self.name,
                        "object_ids": ["G01_1", "G01_2", "G01_3"],
                        "mano_sides": ["right", "left"],
                        "num_frames": 0
                    }

                    self._process_camera(bag_file, output_dir, topics['color'], topics['depth'], metadata)
                    self._generate_metadata(metadata, master_output_dir)
                else:
                    print(f"Camera {camera_id} does not have both color and depth topics.")


    def _generate_metadata(self, metadata, output_dir):
        meta_file = os.path.join(output_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata written to {meta_file}")


def main():
    RosbagExtractor("/home/jikaiwang/GitHub/summer_camp/data/rosbags/may/20240612", "may")
    RosbagExtractor("/home/jikaiwang/GitHub/summer_camp/data/rosbags/lyndon/20240612", "lyndon")
    RosbagExtractor("/home/jikaiwang/GitHub/summer_camp/data/rosbags/nicole/20240612", "nicole")
    print("done")

if __name__ == '__main__':
    main()
