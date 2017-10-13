import torch

from . import base
from . import utils
from .so2 import SO2


class SE2(base.SpecialEuclideanBaseTorch):
    """Homogeneous transformation matrix in SE(2) using active (alibi) transformations."""
    dim = 3
    dof = 3
    RotationType = SO2

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def wedge(cls, xi):
        pass

    @classmethod
    def vee(cls, Xi):
        pass

    @classmethod
    def left_jacobian(cls, xi):
        pass

    @classmethod
    def inv_left_jacobian(cls, xi):
        pass

    @classmethod
    def exp(cls, xi):
        pass

    def log(self):
        pass

    def adjoint(self):
        pass

    @classmethod
    def odot(cls, p, directional=False):
        pass
