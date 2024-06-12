import rosbag
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image

class rosbagExtractor:
    pass

def read_bag_topic_list(bag_file):
    """Reads the list of topics from the bag file."""
    bag = rosbag.Bag(bag_file, 'r')
    topic_list = []
    for topic, _, _ in bag.read_messages():
        if topic not in topic_list:
            topic_list.append(topic)
    bag.close()
    return topic_list



def get_camera_topics(topics):
    """Extract camera IDs and their corresponding color and depth topics."""
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



def save_image(bridge, msg, filename, is_color=True):

    try:
        #COLOR IMAGES
        if is_color:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(filename, cv_img)

        #DEPTH IMAGES
        else:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_array = np.array(cv_img, dtype=np.float32)
            depth_array = depth_array.astype(np.uint16)
            cv2.imwrite(filename, depth_array)

            #cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print("Wrote file: {}".format(filename))

    except CvBridgeError as e:
        print(e)
        print("Error converting image")

    except Exception as e:
        print(f"Unexpected error: {e}")



def process_camera(bag_file, output_dir, color_topic, depth_topic):
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()

    os.makedirs(output_dir, exist_ok=True)

    color_filter = message_filters.SimpleFilter()
    depth_filter = message_filters.SimpleFilter()
    ts = message_filters.ApproximateTimeSynchronizer([color_filter, depth_filter], 10, 0.1, allow_headerless=True)

    color_count = 0
    depth_count = 0

    def callback(color_msg, depth_msg):
        # Keeping track of the color and depth framc counts
        nonlocal color_count, depth_count
        #timestamp = color_msg.header.stamp.to_sec()
        color_filename = os.path.join(output_dir, "color_%06i.jpg" % color_count)
        depth_filename = os.path.join(output_dir, "depth_%06i.png" % depth_count)
        save_image(bridge, color_msg, color_filename, is_color=True)
        save_image(bridge, depth_msg, depth_filename, is_color=False)
        color_count += 1
        depth_count += 1

    ts.registerCallback(callback)

    for topic, msg, _ in bag.read_messages(topics=[color_topic, depth_topic]):
        if topic == color_topic:
            color_filter.signalMessage(msg)
        elif topic == depth_topic:
            depth_filter.signalMessage(msg)

    bag.close()



def process_bag(bag_files):
    for bag_file in bag_files:
        parts = bag_file.split('.')
        bagID = parts[0].split('/')
        master_output_dir = "/home/jikaiwang/GitHub/summer_camp/data/recordings/20231022_193630" + bagID[1] + bagID[2]

        if not os.path.exists(master_output_dir):
            os.makedirs(master_output_dir)

        topics = read_bag_topic_list(bag_file)
        cameras = get_camera_topics(topics)

        for camera_id, topics in cameras.items():
            if 'color' in topics and 'depth' in topics:
                output_dir = os.path.join(master_output_dir, camera_id)
                print(f"Processing camera: {camera_id}")
                process_camera(bag_file, output_dir, topics['color'], topics['depth'])
            else:
                print(f"Camera {camera_id} does not have both color and depth topics.")

def get_bag_files(directory):
    pass



def main():
    rospy.init_node('extract_rosbag', anonymous=True)
    bag_files = ['nicole/20240612/20240612_095906.bag', 'nicole/20240612/20240612_095946.bag', 'nicole/20240612/20240612_100005.bag']
    #processing each of the bag files inside of the directory
    process_bag(bag_files)

if __name__ == '__main__':
    main()
