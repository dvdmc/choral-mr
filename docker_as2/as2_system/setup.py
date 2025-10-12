from setuptools import setup

package_name = 'as2_system'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anonymous',
    maintainer_email='anonym@gmail.com',
    description='PKeep pose node for MAVROS using ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keep_pose = as2_system.keep_pose:main',
        ],
    },
)