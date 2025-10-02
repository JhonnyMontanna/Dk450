from setuptools import find_packages, setup

package_name = 'dk450'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Agregar el launch file para MAVROS
        ('share/' + package_name + '/launch', ['launch/arducopter_mavros.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='drk',
    maintainer_email='drk@todo.todo',
    description='Puente MAVROS para ArduCopter',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MavlinkVisualizer = dk450.MavlinkVisualizer:main',
            'MavrosVisualizer = dk450.MavrosVisualizer:main',
            'Mavros_Circulo = dk450.Mavros_Circulo:main',
            'Mavros_Takeoff = dk450.Mavros_Takeoff:main',
            'Mavros_Land = dk450.Mavros_Land:main',
            'ControlPID_Seguidor = dk450.ControlPID_Seguidor:main',
            'ControlP_Seguidor = dk450.ControlP_Seguidor:main',
        ],
    },
)

