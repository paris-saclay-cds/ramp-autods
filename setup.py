#! /usr/bin/env python

# Copyright (C) 2023 Balazs Kegl

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("rampds", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "ramp-autods"
DESCRIPTION = "Automated tabular data scientist based on the ramp-workflow and ramp-hyperopt libraries."
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "B. Kegl"
MAINTAINER_EMAIL = "balazs.kegl@gmail.com"
URL = "https://github.com/paris-saclay-cds/ramp-autods"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/paris-saclay-cds/ramp-autods"
VERSION = __version__  # noqa
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn>=0.22",
    "joblib",
    "cloudpickle",
    "click",
    "category_encoders",
    "xgboost",
    "lightgbm",
    "catboost",
    "skrub>=0.2",
    "click_config_file",
    "holidays",
    "ramp-workflow @ git+https://github.com/paris-saclay-cds/ramp-workflow@autods",
    "ramp-hyperopt @ git+https://github.com/paris-saclay-cds/ramp-hyperopt@autods",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx_rtd_theme", "numpydoc", "sphinx-click"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "ramp-setup = rampds.cli.setup:start",
            "ramp-hyperopt-race = rampds.cli.hyperopt_race:start",
            "ramp-register-experiment = rampds.cli.register_experiment:start",
            "ramp-blend-at-round = rampds.cli.blend_at_round:start",
            "ramp-foundation-models = rampds.cli.foundation_models:start",
            "ramp-save-results = rampds.cli.save_results:start",
        ]
    },
)
