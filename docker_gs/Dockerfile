# === Select base image: comment/uncomment ONE of the following two lines ===
# For CUDA support:
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
# For plain Ubuntu (no CUDA):
# FROM ubuntu:24.04

# The rest of the Dockerfile continues from here

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

# Initialize rosdep and colcon mixins
RUN rosdep init \
  && rosdep update --rosdistro $ROS_DISTRO \
  && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
  && colcon mixin update \
  && colcon metadata add default https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml \
  && colcon metadata update

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

# For allowing connection between machines, fastdds has some problem.
RUN apt-get update && apt-get install -y ros-jazzy-rmw-cyclonedds-cpp

# Gazebo (Harmonic)
RUN apt-get install -y ros-jazzy-ros-gz

# OptiTrack
RUN mkdir -p /root/optitrack_ws/src
WORKDIR /root/optitrack_ws/src
RUN git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2
RUN git clone -b rolling https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs
RUN git clone https://github.com/MOCAP4ROS2-Project/mocap4ros2_optitrack
WORKDIR /root/optitrack_ws
RUN rosdep install --from-paths src --ignore-src -r -y
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --executor sequential

# AS2 TODO: update to binaries.
RUN apt-get install -y libyaml-cpp-dev
RUN mkdir -p /root/aerostack2_ws/src
RUN git clone -b temporal_fixes https://github.com/dvdmc/aerostack2.git /root/aerostack2_ws/src/aerostack2 && \
    git clone -b main https://github.com/dvdmc/as2_platform_tello.git /root/aerostack2_ws/src/as2_platform_tello
RUN git clone https://github.com/MOCAP4ROS2-Project/mocap4r2_msgs.git /root/aerostack2_ws/src/mocap4r2_msgs
WORKDIR /root/aerostack2_ws
RUN rosdep update && rosdep install -y -r -q --from-paths src --ignore-src

# AS2 dependencies on Nav2
RUN apt update && apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb* -y

RUN source /opt/ros/jazzy/setup.bash && colcon build --parallel-workers 4 --symlink-install

# For Computer Vision DL (requires starting from CUDA)
RUN pip install colcon-lcov-result cpplint cmakelint PySimpleGUI-4-foss --break-system-packages
# Move to the venv for now to make the image smaller
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --break-system-packages
RUN apt install -y libpcl-dev ros-jazzy-pcl-conversions python3.12-venv

## Install OR-TOOLS
WORKDIR /root/
RUN wget https://github.com/google/or-tools/releases/download/v9.11/or-tools_amd64_ubuntu-24.04_cpp_v9.11.4210.tar.gz && \
    tar -xvzf  /root/or-tools_amd64_ubuntu-24.04_cpp_v9.11.4210.tar.gz  && \
    mv /root/or-tools_x86_64_Ubuntu-24.04_cpp_v9.11.4210/ /opt/or-tools && \ 
    rm /root/or-tools_amd64_ubuntu-24.04_cpp_v9.11.4210.tar.gz

WORKDIR /root/
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/optitrack_ws/install/setup.bash" >> /root/.bashrc
RUN echo "source /root/aerostack2_ws/install/setup.bash" >> /root/.bashrc

# Copy ROS entrypoint script
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]