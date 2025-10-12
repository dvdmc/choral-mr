from setuptools import find_packages, setup
import os
from glob import glob

package_name = "sensors_tools_ros"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.xml")),
        (os.path.join("share", package_name, "cfg"), glob("cfg/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anonym@anonym.com",
    maintainer_email="anonym@anonym.com",
    description="Sensors tools for conversion and inference",
    license="BSD",
    extras_require={
    "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "semantic_node = sensors_tools_ros.semantic_node:main",
        ],
    },
)
