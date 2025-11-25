# === Select base image: comment/uncomment ONE of the following two lines ===
# For CUDA support:
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
# For plain Ubuntu (no CUDA):
# FROM ubuntu:24.04

# Install ROS 2 Jazzy
# See https://hub.docker.com/layers/library/ros/jazzy/images/sha256-50a4769126c5ace8e2cf5ba2acacd338fca425149e1ee0bed0cdcb53ffbf429c
RUN echo 'Etc/UTC' > /etc/timezone \
 &&     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime \ 
 &&     apt-get update  \
 &&     apt-get install -q -y --no-install-recommends tzdata \
 &&     rm -rf /var/lib/apt/lists/* # buildkit

RUN apt-get update  \
 && apt-get install -q -y --no-install-recommends ca-certificates curl dirmngr gnupg2 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -L -s -o /tmp/ros2-apt-source.deb https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb \    
  && echo "35441f3092fd05773a3c397fab38661bec466584c7a1f1c05366579997cb5fe7 /tmp/ros2-apt-source.deb" | sha256sum --strict --check \
  && apt-get update \
  && apt-get install /tmp/ros2-apt-source.deb \
  && rm -f /tmp/ros2-apt-source.deb \
  && rm -rf /var/lib/apt/lists/*


ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=jazzy

# Install ROS 2 core and base, colcon, rosdep, and other essentials
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-core=0.11.0-1* \
    ros-jazzy-ros-base=0.11.0-1* \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    ca-certificates \
    curl \
    dirmngr \
    gnupg2 \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y \
  && apt-get install -y \
    tmux \
    tmuxinator \
    vim \
    xterm \
    ssh \
    tree \
    python3-tk \
    nano \
    libtool \
    libtool-bin \
    htop \
    gdb \
    net-tools \
    cmake \
    lsb-release \
    iputils-ping


RUN echo "set -g mouse on" > /root/.tmux.conf

## ROS2 Jazzy utils
RUN apt-get install -y \
    python3-rosdep \
    python3-setuptools \
    python3-pip \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    ros-dev-tools

RUN apt-get install -y \
    python3-flake8 \
    python3-flake8-builtins \
    python3-flake8-comprehensions \
    python3-flake8-docstrings \
    python3-flake8-import-order \
    python3-flake8-quotes

RUN apt-get install -y pylint flake8 pycodestyle
RUN apt-get install -y cppcheck lcov
RUN rosdep init

# For allowing connection between machines, fastdds has some problem.
RUN apt-get install -y ros-jazzy-rmw-cyclonedds-cpp

# Mavlink
RUN apt-get install -y ros-jazzy-mavros
RUN source /opt/ros/jazzy/setup.bash && ros2 run mavros install_geographiclib_datasets.sh

# TODO: update to binaries when AS2 gets updated to Jazzy.
RUN mkdir -p /root/aerostack2_ws/src
RUN git clone -b jazzy-patch-trajgen https://github.com/dvdmc/aerostack2.git /root/aerostack2_ws/src/aerostack2 && \
    git clone -b jazzy https://github.com/dvdmc/as2_platform_mavlink.git /root/aerostack2_ws/src/as2_platform_mavlink
RUN git clone https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs.git /root/aerostack2_ws/src/mocap4r2_msgs
WORKDIR /root/aerostack2_ws
RUN rosdep update && rosdep install -y -r -q --from-paths src --ignore-src

# AS2 dependencies on Nav2
RUN apt-get update && apt-get install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb* -y

# Build with care due to limited resources of Jetson Nano
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install

RUN pip3 install colcon-lcov-result cpplint cmakelint PySimpleGUI-4-foss --break-system-packages

# Install Realsense from: https://github.com/IntelRealSense/realsense-ros. 
# NOTE: Realsense ROS has some issues with the bin release for ARM. We build it for the moment.
RUN apt-get -y install ros-jazzy-librealsense2*
RUN mkdir -p /root/realsense_ws/src
WORKDIR /root/realsense_ws/src
RUN git clone https://github.com/IntelRealSense/realsense-ros.git
WORKDIR /root/realsense_ws
RUN rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
RUN source /opt/ros/jazzy/setup.bash && colcon build

# Install torch for current Jetson version
RUN apt-get install -y python3.12-venv
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --break-system-packages


RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/aerostack2_ws/install/setup.bash" >> /root/.bashrc
RUN echo "source /root/realsense_ws/install/setup.bash" >> /root/.bashrc

# Include a compiled ready version of the current repo
# ADD drone_ws/ /root/drone_ws_compiled
# WORKDIR /root/drone_ws_compiled
# RUN ["/bin/bash", "-c", "colcon build --symlink-install"]

RUN rm -rf log # remove log folder

WORKDIR /root/