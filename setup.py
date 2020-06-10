import os.path
import sys

from setuptools import find_packages

DISTNAME = 'onlikhorn'
DESCRIPTION = "Online Sinkhorn algorithm"
MAINTAINER = 'XXX'
MAINTAINER_EMAIL = 'XXX'
URL = 'XXX'
LICENSE = 'MIT License'
DOWNLOAD_URL = ''
VERSION = '0.1.dev0'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('onlikhorn')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(
        # configuration=configuration,
        packages=find_packages(),
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        zip_safe=False,  # the package can run out of an .egg file
        install_requires=['numpy']
    )
