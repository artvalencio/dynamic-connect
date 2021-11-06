from setuptools import setup
setup(name='dynamic_connect',
version='0.01',
description='Python package for building dynamic functional connectome from EEG',
url='https://github.com/artvalencio/dynamic-connect',
author='Arthur Valencio, IC-Unicamp, RIDC NeuroMat',
author_email='arthuric@unicamp.br',
license='MIT',
packages=['dynamic_connect'],
classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
python_requires='>=3.6',
install_requires=['pandas','numpy','matplotlib','scipy','mne','seaborn',
                  '<cami> @ https://github.com/artvalencio/cami-python/archive/master.zip'],
zip_safe=False)
