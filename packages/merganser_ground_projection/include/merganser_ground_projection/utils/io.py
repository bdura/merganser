import os
import duckietown_utils as dtu

from merganser_ground_projection.exceptions import NoHomographyInfoAvailable


def get_homography_info_config_file(robot_name, strict=False):
    roots = [
        os.path.join(dtu.get_duckiefleet_root(), 'calibrations'),
        os.path.join(dtu.get_ros_package_path('duckietown'),
                     'config', 'baseline', 'calibration')
    ]
    found = []

    for root in roots:
        fn = os.path.join(root, 'camera_extrinsic', '{0}.yaml')
        for name in [robot_name, 'default']:
            filename = fn.format(name)
            if os.path.exists(filename):
                found.append(filename)
                dtu.logger.info('Using filename {0}'.format(filename))

    if not found:
        raise NoHomographyInfoAvailable('Cannot find homography file for robot '
                                        '`{0}` in folders\n{1}'.format(
                                        robot_name, '\n'.join(roots)))
    else:
        if len(found) > 1:
            error_message = ('Found more than one configuration file:\n{0}\n'
                             'Please delete one of those'.format('\n'.join(found)))
            if strict:
                raise Exception(error_message)
            else:
                dtu.logger.error(error_message)
        return found[0]
