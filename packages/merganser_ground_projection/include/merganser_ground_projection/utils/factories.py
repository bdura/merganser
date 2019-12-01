import os
import duckietown_utils as dtu

from pi_camera import get_camera_info_for_robot

from merganser_ground_projection.ground_projection_geometry import GroundProjectionGeometry
from merganser_ground_projection.utils.io import get_homography_info_config_file
from merganser_ground_projection.utils.homography import homography_from_yaml


def get_homography_for_robot(robot_name):
    dtu.check_isinstance(robot_name, str)
    # Get the config file for the homography of the robot
    filename = get_homography_info_config_file(robot_name, strict=False)
    data = dtu.yaml_load_file(filename)

    # Extract the homography from the YAML file
    homography = homography_from_yaml(data)

    return homography

def get_ground_projection_geometry_for_robot(robot_name):
    camera_info = get_camera_info_for_robot(robot_name)
    homography = get_homography_for_robot(robot_name)

    return GroundProjectionGeometry(camera_info, homography)
