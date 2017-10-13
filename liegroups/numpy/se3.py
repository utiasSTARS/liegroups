import numpy as np

from . import base
from .so3 import SO3


class SE3(base.SpecialEuclideanBaseNumpy):
    """Homogeneous transformation matrix in SE(3) using active (alibi) transformations."""
    dim = 4
    dof = 6
    RotationType = SO3

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def is_valid_matrix(cls, mat):
        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[cls.dim - 1, :], np.array([0, 0, 0, 1])) and \
            cls.RotationType.is_valid_matrix(mat[0:cls.dim - 1, 0:cls.dim - 1])

    @classmethod
    def identity(cls):
        return cls.from_matrix(np.identity(cls.dim))

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
            raise ValueError("xi must have length 6")

        rho = xi[0:3]
        phi = xi[3:6]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    def log(self):
        phi = self.RotationType.log(self.rot)
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        rotmat = self.rot.as_matrix()
        return np.vstack(
            [np.hstack([rotmat,
                        self.RotationType.wedge(self.trans).dot(rotmat)]),
             np.hstack([np.zeros((3, 3)), rotmat])]
        )

    @classmethod
    def odot(cls, vec, directional=False):
        """SE(3) \odot operator as defined by Barfoot."""
        vec = np.atleast_2d(vec)
        result = np.zeros([vec.shape[0], vec.shape[1], cls.dof])

        if vec.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless vec is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, 0:3, 0:3] = np.eye(3)

            result[:, 0:3, 3:6] = cls.RotationType.wedge(-vec)

        elif vec.shape[1] == cls.dim:
            # Broadcast magic
            result[:, 0:3, 0:3] = vec[:, 3][:, None, None] * np.eye(3)
            result[:, 0:3, 3:6] = cls.RotationType.wedge(-vec[:, 0:3])

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)
