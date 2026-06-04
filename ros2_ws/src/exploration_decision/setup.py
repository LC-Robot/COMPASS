import os # <<< 新增：导入os库
from glob import glob # <<< 新增：导入glob库
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
        # <<< 核心修改：添加此行以安装launch文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='le', # 您可以修改为您的名字
    maintainer_email='le@example.com', # 您可以修改为您的邮箱
    description='ROS 2 package for exploration decision making.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exploration_coordinator = exploration_decision.exploration_coordinator:main',
        ],
    },
)