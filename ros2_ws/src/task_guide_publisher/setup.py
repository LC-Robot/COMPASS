from setuptools import find_packages, setup

package_name = "task_guide_publisher"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="le",
    maintainer_email="le@example.com",
    description="A simple ROS 2 node to publish a manual task guide target.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "guide_publisher_node = task_guide_publisher.guide_publisher_node:main",
        ],
    },
)
