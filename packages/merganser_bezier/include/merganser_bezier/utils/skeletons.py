import numpy as np

from merganser_msgs.msg import SkeletonMsg


def _extract_skeleton(skeleton):
    cloud = np.array([[point.x, point.y] for point in skeleton.cloud])
    color = skeleton.color

    return cloud, color

def get_yellow_cloud(skeletons):
    cloud = []
    for skeleton in skeletons:
        if skeleton.color == SkeletonMsg.YELLOW:
            for point in skeleton.cloud:
                cloud.append([point.x, point.y])
    return np.asarray(cloud) if cloud else None

def get_white_clouds(skeletons):
    clouds = []
    for skeleton in skeletons:
        if skeleton.color == SkeletonMsg.WHITE:
            clouds.append(np.array([[point.x, point.y] for point in skeleton.cloud]))
    return clouds
