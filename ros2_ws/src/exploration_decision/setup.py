import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'exploration_decision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='le',
    maintainer_email='le@example.com',
    description='ROS 2 package for exploration decision making.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exploration_coordinator = exploration_decision.exploration_coordinator:main',
        ],
    },
)
