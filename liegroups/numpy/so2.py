import numpy as np

from . import base


class SO2(base.SpecialOrthogonalBase):
    """Rotation matrix in SO(2) using active (alibi) transformations."""
    dim = 2
    dof = 1

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def wedge(cls, phi):
        phi = np.atleast_1d(phi)

        Phi = np.zeros([len(phi), cls.dim, cls.dim])
        Phi[:, 0, 1] = -phi
        Phi[:, 1, 0] = phi
        return np.squeeze(Phi)

    @classmethod
    def vee(cls, Phi):
        if Phi.ndim < 3:
            Phi = np.expand_dims(Phi, axis=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(cls.dim, cls.dim, cls.dim, cls.dim))

        return np.squeeze(Phi[:, 1, 0])

    @classmethod
    def left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) + 0.5 * cls.wedge(phi)

        s = np.sin(phi)
        c = np.cos(phi)

        return (1. / phi) * np.array([[s, -(1 - c)],
                                      [1 - c, s]])

    @classmethod
    def inv_left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) - 0.5 * cls.wedge(phi)

        A = np.sin(phi) / phi
        B = (1. - np.cos(phi)) / phi
        return (1. / (A * A + B * B)) * np.array([[A, B], [-B, A]])

    @classmethod
    def exp(cls, phi):
        c = np.cos(phi)
        s = np.sin(phi)

        return cls(np.array([[c, -s],
                             [s,  c]]))

    def log(self):
        c = self.mat[0, 0]
        s = self.mat[1, 0]
        return np.arctan2(s, c)

    def adjoint(self):
        return 1.

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        return cls.exp(angle_in_radians)

    def to_angle(self):
        """Recover the rotation angle in rad from the rotation matrix."""
        return self.log()
