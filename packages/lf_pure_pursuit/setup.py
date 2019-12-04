## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'lf_pure_pursuit',
    ],
    package_dir={
        'lf_pure_pursuit': 'include/lf_pure_pursuit',
    },
)

setup(**setup_args)
