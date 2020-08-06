from setuptools import setup, find_packages

setup(
    name='HP_Servier',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'servier=Servier_package.main:main',
        ]
    }
)