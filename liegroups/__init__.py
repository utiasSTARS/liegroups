"""Special Euclidean and Special Orthogonal Lie groups."""

__all__ = []

import inspect
import pkgutil

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"
