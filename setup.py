from setuptools import setup, find_packages

setup(
    name='debiased_sd',
    version='0.1.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    tests_require=[
        'pytest',
        'numpy',
    ],
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.11.4'
    ],
)