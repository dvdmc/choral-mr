###
#
# This file manages the bridges module.
#
###
from dataclasses import fields
from typing import Literal, Union, TYPE_CHECKING

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .airsim_bridge import AirsimBridge, AirsimBridgeConfig
    from .ros_bridge import ROSBridge, ROSBridgeConfig
    from .scannet_bridge import ScanNetBridge, ScanNetBridgeConfig
    from .scannet_voc_bridge import ScanNetVOCBridge, ScanNetVOCBridgeConfig
    from .test_bridge import TestBridge, TestBridgeConfig
    from .habitat_bridge import HabitatBridge, HabitatBridgeConfig

BridgeConfig = Union[
    "ScanNetBridgeConfig",
    "ScanNetVOCBridgeConfig",
    "TestBridgeConfig",
    "AirsimBridgeConfig",
    "ROSBridgeConfig",
    "HabitatBridgeConfig",
]

BridgeType = Literal[
    "airsim", "ros", "scannet", "scannet_voc", "test", "folder", "habitat"
]

ControllableBridges = [
    "scannet",
    "scannet_voc",
    "test",
    "folder",
    "habitat",
]  # Bridges that can be controlled by the sensor

Bridges = Union[
    "ScanNetBridge",
    "ScanNetVOCBridge",
    "TestBridge",
    "AirsimBridge",
    "ROSBridge",
    "HabitatBridge",
]


def get_bridge_config(bridge_type: BridgeType):
    assert bridge_type is not None, "Bridge type must be specified"

    if bridge_type == "airsim":
        from .airsim_bridge import AirsimBridgeConfig

        return AirsimBridgeConfig
    elif bridge_type == "scannet":
        from .scannet_bridge import ScanNetBridgeConfig

        return ScanNetBridgeConfig
    elif bridge_type == "scannet_voc":
        from .scannet_voc_bridge import ScanNetVOCBridgeConfig

        return ScanNetVOCBridgeConfig
    elif bridge_type == "ros":
        from .ros_bridge import ROSBridgeConfig

        return ROSBridgeConfig
    elif bridge_type == "folder":
        from .folder_bridge import FolderBridgeConfig

        return FolderBridgeConfig
    elif bridge_type == "test":
        from .test_bridge import TestBridgeConfig

        return TestBridgeConfig

    elif bridge_type == "habitat":
        from .habitat_bridge import HabitatBridgeConfig

        return HabitatBridgeConfig

    else:
        raise NotImplementedError("Bridge type not implemented")


def get_bridge(bridge_type: BridgeType, bridge_cfg: BridgeConfig) -> Bridges:
    assert bridge_type is not None, "Bridge type must be specified"
    assert bridge_cfg is not None, "Bridge cfg must be specified"

    if bridge_type == "airsim":
        from .airsim_bridge import AirsimBridge, AirsimBridgeConfig

        assert isinstance(
            bridge_cfg, AirsimBridgeConfig
        ), "Bridge cfg must be of type AirsimBridgeConfig"
        return AirsimBridge(bridge_cfg)
    if bridge_type == "scannet":
        from .scannet_bridge import ScanNetBridge, ScanNetBridgeConfig

        assert isinstance(
            bridge_cfg, ScanNetBridgeConfig
        ), "Bridge cfg must be of type ScanNetBridgeConfig"
        return ScanNetBridge(bridge_cfg)
    elif bridge_type == "scannet_voc":
        from .scannet_voc_bridge import ScanNetVOCBridge, ScanNetVOCBridgeConfig

        assert isinstance(
            bridge_cfg, ScanNetVOCBridgeConfig
        ), "Bridge cfg must be of type ScanNetVOCBridgeConfig"
        return ScanNetVOCBridge(bridge_cfg)
    elif bridge_type == "ros":
        from .ros_bridge import ROSBridge, ROSBridgeConfig

        assert isinstance(
            bridge_cfg, ROSBridgeConfig
        ), "Bridge cfg must be of type ROSBridgeConfig"
        return ROSBridge(bridge_cfg)
    elif bridge_type == "folder":
        from .folder_bridge import FolderBridge, FolderBridgeConfig

        assert isinstance(
            bridge_cfg, FolderBridgeConfig
        ), "Bridge cfg must be of type FolderBridgeConfig"
        return FolderBridge(bridge_cfg)
    elif bridge_type == "test":
        from .test_bridge import TestBridge, TestBridgeConfig

        assert isinstance(
            bridge_cfg, TestBridgeConfig
        ), "Bridge cfg must be of type TestBridgeConfig"
        return TestBridge(bridge_cfg)
    elif bridge_type == "habitat":
        from .habitat_bridge import HabitatBridge, HabitatBridgeConfig

        assert isinstance(
            bridge_cfg, HabitatBridgeConfig
        ), "Bridge cfg must be of type HabitatBridgeConfig"
        return HabitatBridge(bridge_cfg)
    else:
        raise NotImplementedError("Bridge type not implemented")
