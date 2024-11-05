from setuptools import setup, find_packages

setup(
    name='ase-cp2k_noshell',
    version='0.1.0',
    author='Louie Slocombe',
    author_email='louies@hotmail.co.uk',
    description='Code for the ASE CP2K calculator but without using cp2k-shell',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LouieSlocombe/ase-cp2k-noshell',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'ase',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
