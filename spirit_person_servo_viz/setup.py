from setuptools import find_packages, setup

package_name = 'spirit_person_servo_viz'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HJoonKwon',
    maintainer_email='joonk2@andrew.cmu.edu',
    description='Foxglove overlay of the person servo target on the Spirit EO video.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'servo_overlay = spirit_person_servo_viz.servo_overlay_node:main',
        ],
    },
)
