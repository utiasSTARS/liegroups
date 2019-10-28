from setuptools import setup

setup(
    name='liegroups',
    version='2.0.0',
    description='Lie groups in Python',
    author='Lee Clement',
    author_email='lee.clement@robotics.utias.utoronto.ca',
    license='MIT',
    packages=['liegroups', 'liegroups.numpy', 'liegroups.torch'],
    install_requires=['numpy']
)
