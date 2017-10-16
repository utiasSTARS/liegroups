import numpy as np

from . import base
from .so3 import SO3


class SE3(base.SpecialEuclideanBase):
    """Homogeneous transformation matrix in SE(3) using active (alibi) transformations."""
    dim = 4
    dof = 6
    RotationType = SO3

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def wedge(cls, xi):
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dim, cls.dim])
        Xi[:, 0:3, 0:3] = cls.RotationType.wedge(xi[:, 3:7])
        Xi[:, 0:3, 3] = xi[:, 0:3]
        return np.squeeze(Xi)

    @classmethod
    def vee(cls, Xi):
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = np.empty([Xi.shape[0], cls.dof])
        xi[:, 0:3] = Xi[:, 0:3, 3]
        xi[:, 3:6] = cls.RotationType.vee(Xi[:, 0:3, 0:3])
        return np.squeeze(xi)

    @classmethod
    def left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def inv_left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def exp(cls, xi):
        if len(xi) != cls.dof:
            raise ValueError("xi must have length {}".format(cls.dof))

        rho = xi[0:3]
        phi = xi[3:6]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    def log(self):
        phi = self.RotationType.log(self.rot)
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def adjoint(self):
        rotmat = self.rot.as_matrix()
        return np.vstack(
            [np.hstack([rotmat,
                        self.RotationType.wedge(self.trans).dot(rotmat)]),
             np.hstack([np.zeros((3, 3)), rotmat])]
        )

    @classmethod
    def odot(cls, p, directional=False):
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # ptor, in which case the scale is 0
            if not directional:
                result[:, 0:3, 0:3] = np.eye(3)

            result[:, 0:3, 3:6] = cls.RotationType.wedge(-p)

        elif p.shape[1] == cls.dim:
            # Broadcast magic
            result[:, 0:3, 0:3] = p[:, 3][:, None, None] * np.eye(3)
            result[:, 0:3, 3:6] = cls.RotationType.wedge(-p[:, 0:3])

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)
