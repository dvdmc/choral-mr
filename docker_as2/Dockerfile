FROM arm64v8/ros:jazzy-ros-base

SHELL ["/bin/bash", "-c"]

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

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
RUN source /opt/ros/jazzy/setup.bash && MAKEFLAGS="-j2" colcon build --symlink-install --executor sequential

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

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/aerostack2_ws/install/setup.bash" >> /root/.bashrc
RUN echo "source /root/realsense_ws/install/setup.bash" >> /root/.bashrc

# INCLUDE a compiled ready version of the current repo
RUN mkdir -p /root/as2_system_compiled_ws/src
ADD as2_system /root/as2_system_compiled_ws/src/as2_system
WORKDIR /root/as2_system_compiled_ws
RUN ["/bin/bash", "-c", "colcon build --symlink-install"]

RUN rm -rf log # remove log folder

WORKDIR /root/