from setuptools import find_packages, setup

package_name = 'ankle_kinematics'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    # ankle_viz.xml is loaded by viz_mujoco.py relative to __file__, so it has
    # to sit next to the module in the install tree, not only in share/.
    package_data={package_name: ['*.xml']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy>=1.24'],
    zip_safe=True,
    maintainer='eliacriscihuber',
    maintainer_email='eliacriscihuber@gmail.com',
    description='Parallel RSU ankle kinematics: foot pose <-> motor angles. '
                'Pure numpy, no ROS dependency.',
    license='UNLICENSED',
    extras_require={
        'test': ['pytest'],
        'viz': ['matplotlib>=3.7', 'mujoco>=3.0'],
    },
    entry_points={
        'console_scripts': [
            # check_geometry is the one worth having on PATH: it re-validates the
            # constants against the rigid-rod constraint after any CAD change.
            'ankle_check_geometry = ankle_kinematics.check_geometry:main',
        ],
    },
)
