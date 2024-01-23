from setuptools import setup, find_packages

setup(
    name='YOLO-v8-crash-cource',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ultralytics==8.0.26',
        'numpy',
        'matplotlib',
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'your_script_name = YOLO-v8-crash-cource.your_module:main_function',
        ],
    },
    author='viveknaidu007',
    description='yolo-v8',
    url='https://github.com/viveknaidu007/computer-vision-projects.git',
)
