FROM osrf/ros:jazzy-desktop-full

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

# ROS2 Jazzy utils
RUN apt-get install -y \
    python3-setuptools \
    python3-pip \
    ros-dev-tools


# Only for development purposes
# RUN apt-get install -y \
#     python3-flake8 \
#     python3-flake8-builtins \
#     python3-flake8-comprehensions \
#     python3-flake8-docstrings \
#     python3-flake8-import-order \
#     python3-flake8-quotes


# RUN apt-get install -y \
#     pylint \
#     flake8 \
#     pycodestyle \
#     cppcheck \
#     lcov


# For allowing connection between machines, fastdds has some problem.
RUN apt-get update && apt-get install -y ros-jazzy-rmw-cyclonedds-cpp

## Turtlebot2
RUN apt install ros-jazzy-kobuki-velocity-smoother ros-jazzy-sophus -y
RUN mkdir -p /root/turtlebot2_ws/src
WORKDIR /root/turtlebot2_ws/src
# TODO: Fix my forks (dvdmc) when solved!
RUN git clone https://github.com/kobuki-base/kobuki_ros_interfaces.git && \
    git clone https://github.com/kobuki-base/kobuki_core.git && \
    git clone https://github.com/kobuki-base/kobuki_ros.git && \
    git clone https://github.com/stonier/ecl_core.git && \
    git clone https://github.com/dvdmc/ecl_lite.git && \
    git clone https://github.com/kobuki-base/cmd_vel_mux.git
# COPY rules below
# Add the user to dialout group
RUN usermod -a -G dialout root

WORKDIR /root/turtlebot2_ws
RUN rosdep update && rosdep install -i --from-path src --rosdistro jazzy -y
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install --executor sequential

# NAV2
RUN apt update && apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb* -y

# OptiTrack
RUN mkdir -p /root/optitrack_ws/src
WORKDIR /root/optitrack_ws/src
RUN git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2
RUN git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs
RUN git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack
RUN git clone https://github.com/dvdmc/optitrack_pose_broadcaster.git

WORKDIR /root/optitrack_ws
RUN rosdep install --from-paths src --ignore-src -r -y
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install --executor sequential

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/turtlebot2_ws/install/setup.bash" >> /root/.bashrc
RUN echo "source /root/optitrack_ws/install/setup.bash" >> /root/.bashrc

COPY docker/60-kobuki.rules /etc/udev/rules.d/

# INCLUDE a compiled ready version of the current repo
RUN mkdir -p /root/tb2_system_compiled/src
ADD tb2_unizar /root/tb2_system_compiled/src/tb2_system
WORKDIR /root/tb2_system_compiled
RUN ["/bin/bash", "-c", "source /root/optitrack_ws/install/setup.bash && colcon build --symlink-install"]

RUN rm -rf log # remove log folder
WORKDIR /root/