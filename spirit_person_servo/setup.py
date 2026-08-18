from glob import glob

from setuptools import find_packages, setup

package_name = 'spirit_person_servo'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='James Emilian',
    maintainer_email='juvanjos@andrew.cmu.edu',
    description='Onboard person detection and gimbal servoing for Spirit drones.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_servo_node = spirit_person_servo.person_servo_node:main',
            'gimbal_deadman_node = spirit_person_servo.gimbal_deadman_node:main',
            'run_sign_calibration = spirit_person_servo.run_sign_calibration:main',
        ],
    },
)
