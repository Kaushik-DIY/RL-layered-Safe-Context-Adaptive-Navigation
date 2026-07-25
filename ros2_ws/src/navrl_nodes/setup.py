from glob import glob

from setuptools import setup

package_name = "navrl_nodes"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kaushik Mitra",
    maintainer_email="kaushikmitra2310@gmail.com",
    description="Thin ROS 2 wrappers around the sim-agnostic MPC+CBF core (plan D8).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mpc_node = navrl_nodes.mpc_node:main",
            "cbf_node = navrl_nodes.cbf_node:main",
            "human_tracker_node = navrl_nodes.human_tracker_node:main",
        ],
    },
)
