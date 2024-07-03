import rospy
import rosbag
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from .Utils import *


class RosbagExtractor:
    def __init__(self, debug=False) -> None:
        """Initializes the RosbagExtractor object."""
        self._debug = debug

        # Set up paths and logger
        self._recordings_folder = PROJ_ROOT / "data" / "recordings"
        self._logger = get_logger(
            log_level="DEBUG" if debug else "INFO", log_name="RosbagExtractor"
        )

        # Define file naming conventions
        self._color_prefix = "color_{:06d}.jpg"
        self._depth_prefix = "depth_{:06d}.png"

        # Set node ID
        self._node_id = "rosbag_extractor"

        # Initialize the ROS node
        self._init_node()

    def extract_bag(self, bag_path: str, extrinsics_file: str = None) -> None:
        self._logger.info(f"Processing rosbag: {bag_path}")
        self._bag_file = Path(bag_path).resolve()
        self._person_id = self._bag_file.parent.parent.name.lower()
        self._save_folder = (
            self._recordings_folder / f"{self._person_id}_{self._bag_file.stem}"
        )
        make_clean_folder(self._save_folder)

        self._synced_topics = []
        self._synced_messages = []

        try:
            with rosbag.Bag(str(self._bag_file), "r") as bag:
                self._bag_info_dict = bag.get_type_and_topic_info()

                # Get the list of synced topics
                self._logger.info("Creating synced topics...")
                self._synced_topics = self._get_synced_topics(
                    [
                        "/color/image_raw",
                        "/aligned_depth_to_color/image_raw",
                    ]
                )
                if not self._synced_topics:
                    self._logger.error("No synced topics found.")
                    return
                self._logger.info(
                    "Topics to sync:\n  - {}".format("\n  - ".join(self._synced_topics))
                )

                # Sync the messages
                self._logger.info("Syncing messages begin...")
                s_time = time.time()
                self._synced_messages = self._sync_messages(bag)
                e_time = time.time()
                self._serials = set([t.split("/")[1] for t in self._synced_topics])
                self._logger.info(
                    "{} messages synced, {:.3f} seconds used".format(
                        len(self._synced_messages[self._synced_topics[0]]),
                        e_time - s_time,
                    )
                )

                # Generate metadata
                metadata = self._generate_metadata()
                if extrinsics_file:
                    metadata["extrinsics"] = Path(extrinsics_file).parent.name

                # Extract images from the synced messages
                self._logger.info("Extracting synced images...")
                self._extract_synced_images()
                self._logger.info("Images extracted.")

                # Save metadata
                self._logger.info("Saving metadata...")
                write_data_to_json(self._save_folder / "meta.json", metadata)
        except rosbag.ROSBagException as e:
            self._logger.error(f"Error processing rosbag: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")

    def _init_node(self) -> None:
        self._logger.info("Initializing ROS node...")

        # Check if roscore is running
        if not self._is_roscore_running():
            self._logger.error(
                "roscore is not running. Please start roscore and try again."
            )
            return

        # Initialize the node
        try:
            rospy.init_node(self._node_id, anonymous=False)
            self._logger.info("ROS node initialized.")
        except rospy.ROSException as e:
            self._logger.error(f"Failed to initialize ROS node: {e}")
        except Exception as e:
            self._logger.error(
                f"Unexpected exception during ROS node initialization: {e}"
            )

    def _is_roscore_running(self) -> bool:
        try:
            rospy.get_master().getPid()
            return True
        except rospy.ROSException as e:
            self._logger.error(f"ROSException: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected exception: {e}")
        return False

    def _get_synced_topics(self, keywords: List[str]) -> List[str]:
        topics = sorted(
            [k for k in self._bag_info_dict.topics if any([x in k for x in keywords])]
        )
        return topics

    def _sync_messages(self, bag: rosbag.Bag) -> dict:
        synced_messages = {topic: [] for topic in self._synced_topics}

        def callback(*msgs):
            for i, msg in enumerate(msgs):
                synced_messages[self._synced_topics[i]].append(msg)

        fs = [message_filters.SimpleFilter() for _ in self._synced_topics]
        ts = message_filters.ApproximateTimeSynchronizer(
            fs, queue_size=10 * len(self._synced_topics), slop=0.1
        )
        ts.registerCallback(callback)

        for topic, msg, t in bag.read_messages(topics=self._synced_topics):
            fs[self._synced_topics.index(topic)].signalMessage(msg)

        return synced_messages

    def _extract_synced_images(self) -> None:
        def extract_images_by_topic(topic, messages_dict):
            _, serial, img_type = topic.split("/")[:3]
            save_folder = self._save_folder / serial
            save_folder.mkdir(parents=True, exist_ok=True)

            prefix = self._depth_prefix if "depth" in img_type else self._color_prefix

            cv_bridge = CvBridge()
            for idx, msg in enumerate(messages_dict[topic]):
                try:
                    cv_img = cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                except CvBridgeError as e:
                    self._logger.error(f"Failed to convert image: {topic} - {e}")
                    continue

                encoding_to_color_conversion = {
                    "rgb8": cv2.COLOR_RGB2BGR,
                    "rgba8": cv2.COLOR_RGBA2BGR,
                    "bgra8": cv2.COLOR_BGRA2BGR,
                }

                if msg.encoding in encoding_to_color_conversion:
                    cv_img = cv2.cvtColor(
                        cv_img, encoding_to_color_conversion[msg.encoding]
                    )

                cv2.imwrite(str(save_folder / prefix.format(idx)), cv_img)

            del cv_bridge
            del messages_dict[topic]

        tqbar = tqdm(total=len(self._synced_messages), ncols=60, colour="green")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    extract_images_by_topic, topic, self._synced_messages
                ): topic
                for topic in self._synced_messages
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                tqbar.update()
                tqbar.refresh()
        tqbar.close()

    def _generate_metadata(self) -> dict:
        serials = sorted(set([t.split("/")[1] for t in self._synced_topics]))

        first_topic = self._synced_topics[0]
        first_message = self._synced_messages[first_topic][0]

        width = first_message.width
        height = first_message.height
        num_frames = len(self._synced_messages[first_topic])

        metadata = {
            "serials": serials,
            "width": width,
            "height": height,
            "extrinsics": None,
            "mano_calib": self._person_id,
            "object_ids": None,
            "mano_sides": None,
            "num_frames": num_frames,
        }

        return metadata

    def __del__(self) -> None:
        """Destructor that ensures the ROS node is shut down when the object is deleted."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shuts down the ROS node if it is initialized."""
        try:
            if rospy.core.is_initialized():
                rospy.signal_shutdown("Exiting...")
                self._logger.info("ROS node shutdown.")
        except Exception as e:
            self._logger.error(f"Error during ROS shutdown: {e}")
