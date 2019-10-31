from setuptools import setup

setup(
    name='liegroups',
    version='1.1.0',
    description='Lie groups in Python',
    author='Lee Clement',
    author_email='lee.clement@robotics.utias.utoronto.ca',
    license='MIT',
    packages=['liegroups', 'liegroups.numpy', 'liegroups.torch'],
    install_requires=['future', 'numpy']
)
