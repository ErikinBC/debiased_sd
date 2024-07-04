===========
Debiased SD
===========

.. image:: https://img.shields.io/pypi/v/debiased_sd.svg
        :target: https://pypi.python.org/pypi/debiased_sd

.. image:: https://api.travis-ci.com/ErikinBC/debiased_sd.svg
        :target: https://app.travis-ci.com/github/ErikinBC/debiased_sd/

.. image:: https://readthedocs.org/projects/debiased-sd/badge/?version=latest
        :target: https://debiased-sd.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Package for debiasing the sample SD estimator. For more information, see `De-biasing standard deviation estimators <http://www.erikdrysdale.com/sd_debias/>`_. 

* Free software: GNU General Public License v3
* Documentation: https://debiased-sd.readthedocs.io.


Features
--------

* Kurtosis-based estimator
* Bootstrap bias estimator
* Jackknife bias estimator
* Exact Gaussian estimator


Package structure
-----------------

Directory Structure
===================

::

    debiased_sd/
    ├── src/
    │   └── debiased_sd/
    │       ├── __init__.py
    │       ├── estimators.py
    │       ├── utils.py
    │       └── Other source files
    ├── tests/
    │   └── test_debiased_sd.py
    ├── docs/
    │   ├── conf.py
    │   ├── index.rst
    │   └── other_docs_files
    ├── setup.py
    ├── requirements_dev.txt
    └── README.rst



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

PyPI distribution built with Travis CI:

1. travis login --pro --github-token {PRIVATE}
2. travis encrypt "{PRIVATE}" --add deploy.password --com 

