from dataclasses import dataclass, field
import time
from typing import List, Literal, Optional, Tuple
from threading import Lock

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

import habitat_sim #type: ignore
from habitat_sim.utils import common as utils #type: ignore
from habitat_sim.utils import viz_utils as vut #type: ignore
import magnum as mn #type: ignore

from sensors_tools.base.cameras import CameraData
from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

HabitatSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class HabitatBridgeConfig(BaseBridgeConfig):
    """
        Configuration class for HabitatBridge
    """
    data_types: List[HabitatSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "pose"]})
    """ Data types to query """

    scene: str = "0"
    """ Scene path """

    enable_physics: bool = True
    """ Enable physics """

    default_agent: int = 0
    """ Index of the default agent """

    width: int = 512
    """ Image width """

    height: int = 512
    """ Image height """

    hfov: float = 90.0
    """ Horizontal field of view """

    sensor_height: float = 1.5
    """ Height of the sensor in meters, relative to the agent"""

    display: bool = False
    """ Display the simulator """

    make_video: bool = False
    """ Make video of the simulation """

class HabitatBridge(BaseBridge):
    """
        Bridge for Habitat
    """
    def __init__(self, cfg: HabitatBridgeConfig):
        """
            Constructor
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.any_sensor = False
        self.ready = False

    def setup(self):
        """
            Setup the bridge
        """
        # Data acquisition configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.cfg.scene
        sim_cfg.gpu_device_id = 0
        sim_cfg.enable_physics = self.cfg.enable_physics

        # Note: all sensors must have the same resolution
        sensor_specs = []
        self.cfg.data_sensor_dict = {}
        if "rgb" in self.cfg.data_types:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            self.cfg.data_sensor_dict["rgb"] = color_sensor_spec.uuid
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [self.cfg.height, self.cfg.width]
            color_sensor_spec.hfov = self.cfg.hfov
            color_sensor_spec.position = [0.0, self.cfg.sensor_height, 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)
            self.any_sensor = True

        if "depth" in self.cfg.data_types:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            self.cfg.data_sensor_dict["depth"] = depth_sensor_spec.uuid
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.cfg.height, self.cfg.width]
            depth_sensor_spec.hfov = self.cfg.hfov
            depth_sensor_spec.position = [0.0, self.cfg.sensor_height, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)
            self.any_sensor = True

        if "semantic" in self.cfg.data_types:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            self.cfg.data_sensor_dict["semantic"] = semantic_sensor_spec.uuid
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [self.cfg.height, self.cfg.width]
            semantic_sensor_spec.hfov = self.cfg.hfov
            semantic_sensor_spec.position = [0.0, self.cfg.sensor_height, 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec)
            self.any_sensor = True

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        # TODO: action space usage is to be determined

        # Habitat cfg
        self.habitat_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

        # try:  # Needed to handle out of order cell run in Jupyter
        #     self.sim.close()
        # except NameError:
        #     pass
        self.sim = habitat_sim.Simulator(self.habitat_cfg)

        # TODO: Add pathfinder: 
        # the navmesh can also be explicitly loaded
        # sim.pathfinder.load_nav_mesh(
        #     os.path.join(data_path, "scene_datasets/habitat-test-scenes/apartment_1.navmesh")
        # )

        self.agent = self.sim.initialize_agent(self.cfg.default_agent)
        
        # Get camera info from the simulator
        if self.any_sensor:
            self.camera_info = CameraData.from_fov_h(self.cfg.width, self.cfg.height, self.cfg.hfov)

        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
        self.agent.set_state(agent_state)

        self.ready = True

    def process_img_responses(
        self, responses: dict
    ) -> dict:
        """
        Process the data from Habitat.
        """
        img_data = {}

        for key, response in responses.items():
            if key == "color_sensor":
                img_data["rgb"] = response # TODO: check size and format

            elif key == "depth_sensor":
                img_data["depth"] = response

            elif key == "semantic_sensor":
                img_data["semantic"] = response

        return img_data
    
    def get_data(self) -> dict:       
        """
            Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            if self.any_sensor:
                if "rgb" in self.cfg.data_types:
                    agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["rgb"]]
                elif "depth" in self.cfg.data_types:
                    agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["depth"]]
                elif "semantic" in self.cfg.data_types:
                    agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["semantic"]]
            else:
                agent_state = self.agent.get_state()
                agent_state.position += np.array([0.0, 0.0, self.cfg.sensor_height])

            print(agent_state)
            translation = np.array([agent_state.position[0], agent_state.position[1], agent_state.position[2]])
            # NOTE: Habitat uses quaterion library with format (w, x, y, z). Scipy rotation uses (x, y, z, w)
            quat = np.array([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
            rotation = Rotation.from_quat(quat)

        if self.any_sensor:
            observations = self.sim.get_sensor_observations(agent_ids=[self.cfg.default_agent])
            img_data = self.process_img_responses(observations)

        data.update(img_data)

        return data
    
    def get_pose(self) -> Tuple[np.ndarray, Rotation]:
        """
            Get pose from the bridge
        """
        if self.any_sensor:
            if "rgb" in self.cfg.data_types:
                agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["rgb"]]
            elif "depth" in self.cfg.data_types:
                agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["depth"]]
            elif "semantic" in self.cfg.data_types:
                agent_state = self.agent.get_state().sensor_states[self.cfg.data_sensor_dict["semantic"]]
        else:
            agent_state = self.agent.get_state()
            agent_state.position += np.array([0.0, 0.0, self.cfg.sensor_height])

        print(agent_state)
        translation = np.array([agent_state.position[0], agent_state.position[1], agent_state.position[2]])
        # NOTE: Habitat uses quaterion library with format (w, x, y, z). Scipy rotation uses (x, y, z, w)
        quat = np.array([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
        rotation = Rotation.from_quat(quat)
        return (translation, rotation)
    
    def move_to_pose(self, traslation: np.ndarray, rotation: Rotation):
        """
            Move the agent to a new pose
        """
        agent_state = self.agent.get_state()
        agent_state.position = traslation
        quat = rotation.as_quat(canonical=False)
        agent_state.rotation = np.array([quat[3], quat[0], quat[1], quat[2]])
        self.agent.set_state(agent_state)

        time.sleep(0.1)
        return True