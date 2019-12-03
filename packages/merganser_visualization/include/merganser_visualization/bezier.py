import numpy as np

from visualization_msgs.msg import MarkerArray

from merganser_visualization.general import color_to_rgba, line_to_marker
from merganser_bezier.utils.bernstein import compute_curve


def bezier_to_marker(bezier, name, veh_name='default', color=0):
    controls = np.asarray([[vector.x, vector.y] for vector in bezier.controls])
    curve = compute_curve(controls, n=20)
    marker = line_to_marker(curve,
                            color_to_rgba(color),
                            name='beizer_curves',
                            veh_name=veh_name)

    return marker

def beziers_to_marker_array(beziers, veh_name='default'):
    markers = []
    beziers = [
        b for b in [beziers.left, beziers.yellow, beziers.right]
        if b.fitted
    ]
    colors = [0, 1, 0]
    for index, bezier in enumerate(beziers):
        marker = bezier_to_marker(bezier, index, veh_name=veh_name, color=colors[index])
        marker.id = index
        markers.append(marker)

    marker_array = MarkerArray()
    marker_array.markers = markers

    return marker_array
