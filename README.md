# CHORAL: A General Framework for Safe Heterogeneous Multi-robot Coordination in Inspection Tasks

<p align="center">
  <img src="media/teaser.png" alt="Main image" width="200" style="margin-right: 40px;">
  <img src="media/teaser.png" alt="Main image" width="200">
</p>

This repository includes te code for the paper _"CHORAL: A General Framework for Safe Heterogeneous Multi-robot Coordination in Inspection Tasks"_ under review for IEEE Robotics & Automation Letters.

The system is composed of: a routing and coordination module, an environmental awareness acquisition module, and the individual robot navigation modules for conducting the mission. The system is fully containerized to allow easy deployment in a general and practical manner. This repository allows the execution of all the software components to further validate our contributions, and provide instructions and deployment configurations for the hardware components, which we believe will be useful for multi-robot deployment settings. 

**NOTE: Since we do not expect reviewers to run a real deployment, some components of the platform-specific Dockerfiles have been removed for anonymity purposes.**

## Repository structure

This repository contains:
- `docker-compose{-tb2/as2}.yaml` are the entrypoints for running the system on the ground station, ground robot, or aerial platform. They define the Dockerfiles to be used to run the system in each platform, define environmental variables and map volumes to the host machine.
- `docker_gs/` folder including the ground station Dockerfile.
- `docker_tb2/`folder including the Turtlebot Dockerfile and the ROS 2 packages `tb2_system` and `optitrack_pose_broadcaster` required to run navigation on the ground robot.
- `docker_as2/` folder including the Surveyor Dockerfile and the ROS 2 package `as2_system` required to run navigation on the surveyor drone.
- `bags`, `maps`, and `results` are empty folers to map volumes inside the ground station Docker to collect persistent data.
- `docker_gs/mr_ws` workspace including different packages used in the ground station. Specifically:
  - `sensors_tools` our package used for inference on images. It is a general module, and the `Trident` configuration was used.
  - `Bloomxai` our 3D semantic voxel-based map software.
  - `het_vrp` the planning and coordination package implementing our heterogeneous VRP formulation, the cost computation, and PRM construction.

## Installation

All the system has been dockerized to ease the installation process. However, any user could follow the installation steps in the Dockerfiles to install the whole system natively.

For configuring Docker, follow the [official installation procedure](
https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) for Ubuntu. Make sure to finish the post-installation steps for running Docker without `sudo` or add it to the commands from here. See below for configuring GPU compatibility (optional).

Once Docker (and optionally NVIDIA Container Toolkit) is installed and validated, you can build the different containers with `docker compose`, which will use the `docker-compose.yaml` file by default. Docker compose is installed by default in modern Docker versions. Thus, you can simply do:

```bash
docker compose build
```

The Docker setup has been tested on Ubuntu 24.04 for the ground station and Ubuntu 20.04 for the companion computers.
The other `docker-compose-{tb/as2}.yaml` files are built inside the Turtlebot and Surveyor drone companion computers' respectively using:

```bash
docker compose -f docker-compose-{tb/as2}.yaml build
```

The Turtlebot's companion computer is an Intel NUC connected to the base via USB. The Surveyor drone's companion computer is an Nvidia Jetson Nano set up with JetPack 4.6 JetPack 4.6.6 [L4T 32.7.6] and a Pixhawk with firmware version v1.14.

### Configuring GPU usage (optional)

If you want to run the environmental awareness module, you need to install CUDA and the NVIDIA Container Toolkit. 

#### Nvidia drivers

First, **only** if nvidia drivers are not installed, we recommend installing with:

```bash
sudo ubuntu-drivers autoinstall
```

#### CUDA

Next, if you also have to install CUDA, we recommend CUDA `12.8` (other `>12` might also work). Follow the official guide for [installing CUDA](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local). Add the following to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

#### Nvidia Container Toolkit

Finally, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian) to enable the use of GPU inside Docker. Remember to validate the installation by importing Pytorch and checking the compute capabilities with `print(torch.cuda.is_available())` within some virtual environment where torch is installed (see Ground Station if required).

## Running demo

To run the virtual example after building the Dockerfile for the ground station, you can run:

```bash
xhost +
docker compose up -d
docker exec -it mr_system /bin/bash
```

Once you are inside the Docker, run:
```bash
cd mr_ws
colcon build --symlink-install
source install/setup.bash
tmuxinator start -p launch-gs_virtual.yaml
# On the terminal that appears
mr_coord_virtual.launch num_agents:=2 map:=forest 
```

You can modify the number of agents to 1 and 4. Other numbers require creating specific config files in `mr_ws/src/het_vrp/mr_het_coord_ros/cfg`. You can also specify other map names.

## Real robots deployment

The real robot deployment highly depends on the specific lab setup and configurations. In the following we offer a comprehensive configuration guide for our different components. Even though specific steps might be required in other systems, we believe that this guide will ease the adoption of our multi-robot system.

### Ground Station

The ground station will run the positioning system for the motion capture system. This should be replaced by individual robot positioning systems in other scenarios. For the motion capture system, adjust the `config/ground_station/mocap4r2_optitrack.yaml` file to set the correct IPs for connecting to the Optitrack system. The ground station also runs semantic segmentation inference on images received from the Surveyor drone. Configure the open-vocabulary terms to detect in `src/sensors_tools/sensors_tools_ros/cfg/sensor_rs_open_trident.yaml`. Aside from building the main source code, you will have to create a virtual environment and install the `sensors_tools` package. Inside the docker, move to `mr_ws` and do:

```bash
cd mr_ws
python3 -m venv .env-sem
source .env-sem/bin/activate
pip install torch torchvision torchaudio # Make sure the CUDA version is installed
cd src/sensors_tools
pip install -e .
```

Now, make sure to being sourced in the venv and rebuild the ws:

```bash
source .env-sem/bin/activate
pip install rospkg lark
pip install -U colcon-common-extensions
python3 -m colcon build --symlink-install
```

NOTE: the last step is a [known issue](https://github.com/ros2/ros2/issues/1094#issuecomment-2897480048) in colcon & ROS 2 when running venvs and we hope to solve it in the future.

### Tello

The Tello drone will be controlled from the ground station since it lacks any companion computer. Please, configure the `config/tello/config.yaml` to set the IP to connect to and set `rigid_body_name` to the rigid body ID.

### Mavlink

The drone used as a Surveyor is commanded through mavlink using a companion computer. It uses Aerostack2 following the [project_mavlink](https://github.com/aerostack2/project_mavlink) approach. Please, configure inside your drone the `config/config.yaml` to modify the following:
- Since we are using Optitrack, set `rigid_body_name` to the rigid body ID.
- Match your specific PX4 / Mavros configuration.

## Citation

Please, if you use the code in this repository, cite the original source.

## Acknowledgements

We would like to thank the work of Nav2 and Aerostack2 ROS 2 packages. We would also like to thank Trident authors for open-sourcing their implementation.
