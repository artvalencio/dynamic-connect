from setuptools import setup, find_packages
setup(name='dynamic_connect',
version='0.01',
description='Python package for building dynamic functional connectome from EEG',
url='https://github.com/artvalencio/dynamic-connect',
author='Arthur Valencio, IC-Unicamp, RIDC NeuroMat',
author_email='arthuric@unicamp.br',
license='MIT',
packages=find_packages(),
classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
python_requires='>=3.6',
install_requires=['pandas','numpy','matplotlib','scipy','mne','seaborn',
                  'cami @ git+https://github.com/artvalencio/cami-python.git@master',
                  ],
zip_safe=False)
