# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup script to build Data-Services."""

import os
import setuptools
from release.python.utils import utils

PACKAGE_LIST = [
    "nvidia_tao_ds"
]

version_locals = utils.get_version_details()
setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))

if(os.path.exists("pytransform_vax_001219")):
    pyarmor_packages = ["pytransform_vax_001219"]
    setuptools_packages += pyarmor_packages

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.py', "*.pyc", "*.yaml", "*.so", "*.pdf", "*.npy", "*.pt", "*.cpp", "*.cu"]
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'augmentation=nvidia_tao_ds.augmentation.entrypoint.augment:main',
            'auto_label=nvidia_tao_ds.auto_label.entrypoint.auto_label:main',
            'annotations=nvidia_tao_ds.annotations.entrypoint.annotations:main',
            'analytics=nvidia_tao_ds.data_analytics.entrypoint.analytics:main',
            'image=nvidia_tao_ds.image.entrypoint.image:main',
            'gap_analysis=nvidia_tao_ds.rcca.gap_analysis.entrypoint.gap_analysis:main',
            'tmm=nvidia_tao_ds.mining.tmm.entrypoint.tmm:main',
            'embedding=nvidia_tao_ds.mining.embedding.entrypoint.embedding:main'
        ]
    }
)
