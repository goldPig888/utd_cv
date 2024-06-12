# ROS Environment Setup on Local Desktop with Conda

ref: [RoboStack Installation](https://robostack.github.io/GettingStarted.html).

# Install Miniconda if you haven't

Follow the official [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html) to install Miniconda.

# Install mamba in the base environment

```bash
conda install mamba -c conda-forge
```

# Install ROS1 into the environment

Because the ROS1 only supports Python 3.9, we need to create a new conda environment with Python 3.9.

- Create a new conda environment with Python 3.9

```bash
mamba create -n ros python=3.9
```

- Activate the new environment

```bash
conda activate ros
```

- Install ROS1

```bash
# adds the conda-forge channel to the new created environment configuration
conda config --env --add channels conda-forge

# and the robostack channel
conda config --env --add channels robostack-staging

# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# install ROS1
mamba install ros-noetic-desktop
```

- Reactivate the environment to initialize the ros env

```bash
mamba deactivate
mamba activate ros
```

- Testing installation

After installation you are able to run rviz and other ros tools.

  - First terminal: start roscore
    ```bash
    mamba activate ros_env
    roscore
    ```

  - Second terminal: start rviz
    ```bash
    mamba activate ros_env
    rviz
    ```
    ![terminal](./resources/ros_installation_testing.png)
    ![rviz](./resources/ros_installation_rviz.png)

