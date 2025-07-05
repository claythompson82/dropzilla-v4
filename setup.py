from setuptools import setup, find_packages

setup(
    name='dropzilla',
    version='4.0.0-alpha',
    author='Clay & 82Edge Trading Ops',
    author_email='ops@82edge.com',
    description='Dropzilla v4: An Institutional-Grade Intraday Signal Engine',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/82-edge/dropzilla-v4', # Replace with your private repo URL
    packages=find_packages(exclude=['tests*', 'scripts*']),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Private :: Do Not Upload',
    ],
    python_requires='>=3.10',
)
