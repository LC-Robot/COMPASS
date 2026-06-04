from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'detect_graspnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='le',
    maintainer_email='le@example.com',
    description='Detection and grasp integration helpers for COMPASS.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_server = detect_graspnet.grasp_server_node:main',
        ],
    },
)
