from setuptools import setup, find_packages

setup(
    name="serl_robot_infra",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "opencv-python",
        "pyquaternion",
        "scipy",
        "pyspacemouse",
        "hidapi",
    ],
    extras_require={
        "franka": [
            "pyyaml",
            "pymodbus==2.5.3",
            "pyrealsense2",
            "rospkg",
            "requests",
            "flask",
            "defusedxml",
        ]
    },
)
