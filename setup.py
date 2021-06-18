import os.path as osp
from setuptools import setup, find_packages

# todo: modify as we decide on versions, names, descriptions. readme
MIN_PY_VER = '3.7'
DISTNAME = 'recOrder'
DESCRIPTION = 'computational microscopy toolkit for label-free imaging'
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()
    LONG_DESCRIPTION_content_type = "text/markdown"
    # LONG_DESCRIPTION = __doc__
LICENSE = 'Chan Zuckerberg Biohub Software License'
DOWNLOAD_URL = ''

INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib', ]
REQUIRES = []

# todo: modify for python dependency
CLASSIFIERS = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Utilities',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

# populate packages
PACKAGES = [package for package in find_packages()]

# parse requirements
with open(osp.join('requirements', 'default.txt')) as f:
    requirements = [line.strip() for line in f
                    if line and not line.startswith('#')]

# populate requirements
for l in requirements:
    sep = l.split(' #')
    INSTALL_REQUIRES.append(sep[0].strip())
    if len(sep) == 2:
        REQUIRES.append(sep[1].strip())

# todo: modify console entry point
if __name__ == '__main__':
    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_content_type,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version="0.0.1",
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        python_requires=f'>={MIN_PY_VER}',
        packages=PACKAGES,
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'recOrder.reconstruct = recOrder.cli_module:main'
            ]
        }
    )
