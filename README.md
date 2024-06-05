# Summer Camp

## Install Git

- Linux

  ```bash
  sudo apt-get install git
  ```

- Windows

  I suguest to use [Github Desktop](https://desktop.github.com/), which is more user-friendly.

- MacOS

  You can install git either via [Homebrew](https://brew.sh/), or [Github Desktop](https://desktop.github.com/).


## Clone the Repository
  
  ```bash
  git clone https://github.com/gobanana520/summer_camp.git summer_camp
  ```

## Python Environment Setup

Follow steps in the [Python Environment Setup](./docs/Python_Environment_Setup.md) document to setup your Python environment.

### Practices
- [Pythion Basics](./notebooks/01_Python_Basics.ipynb)

  Try to understand basics in Python, such as list, tuple, set, dictionary, class, function, loop, etc.
- [Numpy Basics](./notebooks/02_Python_Numpy.ipynb)

  Try to understand basics in Numpy, such as array, matrix, operation, etc.
- CV Basics
  - [Deprojection](./notebooks/03-1_CV_Deprojection.ipynb)

    Try to understand the deprojection of 2D image points to 3D world points.

  - [Triangulation](./notebooks/03-2_CV_Triangulation.ipynb)

    Try to understand the triangulation of 2D image points to 3D world points.

- RANSAC Algorithm
  - [RANSAC](./notebooks/04_RANSAC_Algorithm.ipynb)

    Try to understand the RANSAC algorithm and implement it.

## Reading List

If you are interested in computer vision and finished the practices, you can start reading the following papers and instructions.

### RANSAC Algorithm
- [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)

### SDF (Signed Distance Function)
- [Wiki](https://en.wikipedia.org/wiki/Signed_distance_function)

### Human Body Models
- Papers
  - [SMPL Body Model](./docs/papers/SMPL.pdf)
  - [SMPL-H Body Model](./docs/papers/SMPL-H.pdf)
  - [SMPL-X Body Model](./docs/papers/SMPL-X.pdf)

### Object Pose Estimation
- FoundationPose
  - [Paper](./docs/papers/FoundationPose.pdf)
  - [Website](https://nvlabs.github.io/FoundationPose)

### Image Segmentation
- Segment Anything
  - [Paper](./docs/papers/SegmentAnything.pdf)
  - [Website](https://github.com/facebookresearch/segment-anything)

### 2D Handmarks Detector
- [MediaPipe Handmarks Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

### Video Object Segmentation
- XMem
  - [Paper](./docs/papers/XMem.pdf)
  - [Website](https://hkchengrex.com/XMem)

