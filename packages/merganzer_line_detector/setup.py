## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'merganzer_line_detector',
        'merganzer_line_detector.utils'
    ],
    package_dir={
        'merganzer_line_detector': 'include/merganzer_line_detector',
        'merganzer_line_detector.utils': 'include/merganzer_line_detector/utils'
    },
)

setup(**setup_args)
