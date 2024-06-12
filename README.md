# Summer Camp

## Install Git

- Linux

  ```bash
  sudo apt-get install git
  ```

- Windows

  I suguest to use [Github Desktop](https://desktop.github.com/), which is more user-friendly. Otherwise, you could install git via [Git for Windows](https://gitforwindows.org/).

- MacOS

  You can install git either via [Homebrew](https://brew.sh/), or [Github Desktop](https://desktop.github.com/).

  - Homebrew
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

## Install the Code Editor (VSCode for example)

- You could install the Visual Studio Code (VSCode) from the [official website](https://code.visualstudio.com/).
- Once you have installed the VSCode, you could install below extensions:
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
  - [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
  - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
  - [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)


## Clone the Repository
  
  ```bash
  git clone https://github.com/gobanana520/summer_camp.git summer_camp
  ```

## Python Environment Setup

Follow steps in the [Python Environment Setup](./docs/Python_Environment_Setup.md) document to setup your Python environment.

---

## Project Schedule

### Week 1
- [Pythion Basics](./notebooks/01_Python_Basics.ipynb)
  Try to understand basics in Python, such as list, tuple, set, dictionary, class, function, loop, etc.

- [Numpy Basics](./notebooks/02_Python_Numpy.ipynb)
  Try to understand basics in Numpy, such as array, matrix, operation, etc.

- [Pytorch Basics](./notebooks/06-1_Pytorch_Basics.ipynb)
  Try to understand basics in Pytorch, such as tensor, operation, etc.
- [01_ComputerVisionBasics](./docs/slides/01_ComputerVisionBasics.pdf)
  - Practice 1: [Transformation](./notebooks/03-3_CV_Transformation.ipynb)
    How to apply the transformation to 3D points.
  - Practice 2: [Deprojection](./notebooks/03-1_CV_Deprojection.ipynb)
    How to depreject the 2D image points to 3D camera points.
  - Practice 3: [Triangulation](./notebooks/03-2_CV_Triangulation.ipynb)
    How to calculate the 3D world points from 2D image points.
  - Homework 1: [SequenceLoader](./notebooks/hw1_SequenceLoader.ipynb)
    Write a class to load the data from sequence recording.
- [02_Introduction_to_ROS](./docs/slides/02_Introduction_to_ROS.pdf)
  Understand the basic concepts and useful commands in ROS.
- [03_Introduction_to_MANO](./docs/slides/03_Introduction_to_MANO.pdf)
  Understand the basic concepts of parametric hand model MANO. And introduce the Pytorch implementation of MANO ([Manopth](https://github.com/hassony2/manopth)).
  - Practice 4: [MANO_Hand](./notebooks/05_MANO_Hand.ipynb)
    Understand how to initialize the MANO layer and run the forward process.
- [04_Introduction_to_Optimization](./docs/slides/04_Introduction_to_Optimization.pdf)
  Understand the basic concepts of optimization and the optimization algorithms.
  - Practice 5: [Optimization](./notebooks/06-2_MANO_Pose_Optimization.ipynb)
    Implement the optimization algorithm to optimize the MANO hand pose parameters to fit the target 3D keypoints.
- Readings
  - Highlights
    - [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
      - Practice: [RANSAC](./notebooks/04_RANSAC_Algorithm.ipynb)
        A simple implementation of RANSAC algorithm.
    - [SDF (Signed Distance Function)](https://en.wikipedia.org/wiki/Signed_distance_function)
        Understand the basic concept of SDF. We will use SDF loss to optimize the Hand/Object pose.
    - ROS message synchronization & extraction
      - [Export image from rosbag](https://gist.github.com/zxf8665905/2d09d25da823b0f7390cab83c64d631a)
        Understand how to synchronize the messages from different topics, and extract the images. We will write a RosbagExtractor to extract the images from the rosbag recordings.
    - [MediaPipe Handmarks Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
      Understand the MediaPipe Handmarks Detection. We will write the HandmarksDetector to detect the handmarks from the images.
  - Papers (Optional)
    - [SMPL Body Model](./docs/papers/SMPL.pdf)
    - [SMPL-H Body Model](./docs/papers/SMPL-H.pdf)
    - [SMPL-X Body Model](./docs/papers/SMPL-X.pdf)
  - Methods will be used in the project
    - Object Pose Estimation
      - [FoundationPose](https://nvlabs.github.io/FoundationPose)
    - Image Segmentation
      - [Segment Anything](https://github.com/facebookresearch/segment-anything)
    - Video Object Segmentation
      - [XMem](https://hkchengrex.com/XMem)

### Week 2

- Overview of FoundationPose and Segment Anything
  - [FoundationPose](./docs/papers/FoundationPose.pdf)
  - [Segment Anything](./docs/papers/SegmentAnything.pdf)
- Camera Calibration for the latest Canera Extrinsics
  ![camera_calibration](./docs/resources/camera_calibration_vicalib.gif)
- Hand Calibration for each team members.
- Record data with ROS
  - Select the Objects
  - Design the tasks (e.g., pick and place, handover, etc.)
- Extract the images from the rosbag recordings
- **Homeworks**
  - HW1: Rosbag_Extraction
    - Try to write the class `RosbagExtractor` 
      - to extract the images from the rosbag recordings for all the camera image topics.
      - the extracted images should be saved in the `./data/recordings` folder following below structure
        ```
        20231022_193630           # the rosbag name
        ├── 037522251142          # the camera serial number
        │   ├── color_000000.jpg  # the color image color_xxxxxx.jpg
        │   └── depth_000000.png  # the depth image depth_xxxxxx.png
        │   └── ...  
        ├── 043422252387
        │   ├── color_000000.jpg
        │   ├── depth_000000.png
        │   ├── ...
        ├── ...
        ├── 117222250549
        │   ├── color_000000.jpg
        │   ├── depth_000000.png
        │   ├── ...
        ```
    - The demo rosbag file could be downloaded [here](https://utdallas.box.com/s/uaraafw1mmuofc017vjejxtzdprpvhmv).
    - If you plan to run the ROS locally, you could follow the [ROS Environment Setup](./docs/ROS_Environment_Setup.md) document to setup the ROS environment with conda. Then you could run the `roscore` to start the ROS master, and debug your code under the ROS environment.
    - References:
      - [Export image from rosbag](https://gist.github.com/zxf8665905/2d09d25da823b0f7390cab83c64d631a)

### Week 3

**<code style="color : Red">TBD</code>**