## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'merganser_line_detector',
        'merganser_line_detector.utils'
    ],
    package_dir={
        'merganser_line_detector': 'include/merganser_line_detector',
        'merganser_line_detector.utils': 'include/merganser_line_detector/utils'
    },
)

setup(**setup_args)