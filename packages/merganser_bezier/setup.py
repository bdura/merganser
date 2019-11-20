## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'merganser_bezier',
        'merganser_bezier.utils'
    ],
    package_dir={
        'merganser_bezier': 'include/merganser_bezier',
        'merganser_bezier.utils': 'include/merganser_bezier/utils'
    },
)

setup(**setup_args)
