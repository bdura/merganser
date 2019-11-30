import numpy as np
import yaml
import duckietown_utils as dtu

from merganser_ground_projection.exceptions import InvalidHomographyInfo


def homography_from_yaml(data):
    try:
        homography = data['homography']
        result = np.array(homography).reshape((3, 3))
        return result

    except Exception as e:
        message = 'Could not interpret data:'
        message += '\n\n' + dtu.indent(yaml.dump(data), '   ')
        dtu.raise_wrapped(InvalidHomographyInfo, e, message)
