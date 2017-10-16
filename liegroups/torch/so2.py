import torch

from . import base
from . import utils


class SO2(base.SpecialOrthogonalBase):
    """Rotation matrix in SO(2) using active (alibi) transformations."""
    dim = 2
    dof = 1

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def wedge(cls, phi):
        Phi = phi.__class__(phi.shape[0], cls.dim, cls.dim).zero_()
        Phi[:, 0, 1] = -phi
        Phi[:, 1, 0] = phi
        return Phi.squeeze_()

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi = Phi.unsqueeze(dim=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(cls.dim, cls.dim, cls.dim, cls.dim))

        return Phi[:, 1, 0].squeeze_()

    @classmethod
    def left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                + 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            s = phi[large_angle_inds].sin()
            c = phi[large_angle_inds].cos()

            A = s / phi[large_angle_inds]
            B = (1. - c) / phi[large_angle_inds]

            jac_large_angle = phi.__class__(
                len(large_angle_inds), cls.dim, cls.dim)
            jac_large_angle[:, 0, 0] = A
            jac_large_angle[:, 0, 1] = -B
            jac_large_angle[:, 1, 0] = B
            jac_large_angle[:, 1, 1] = A
            jac[large_angle_inds] = jac_large_angle

        return jac.squeeze_()

    @classmethod
    def inv_left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                - 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            s = phi[large_angle_inds].sin()
            c = phi[large_angle_inds].cos()

            A = s / phi[large_angle_inds]
            B = (1. - c) / phi[large_angle_inds]
            C = (1. / (A * A + B * B))

            jac_large_angle = phi.__class__(
                len(large_angle_inds), cls.dim, cls.dim)
            jac_large_angle[:, 0, 0] = C * A
            jac_large_angle[:, 0, 1] = C * B
            jac_large_angle[:, 1, 0] = -C * B
            jac_large_angle[:, 1, 1] = C * A
            jac[large_angle_inds] = jac_large_angle

        return jac.squeeze_()

    @classmethod
    def exp(cls, phi):
        s = phi.sin()
        c = phi.cos()

        mat = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c

        return cls(mat.squeeze_())

    def log(self):
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        s = mat[:, 1, 0]
        c = mat[:, 0, 0]

        return torch.atan2(s, c).squeeze_()

    def adjoint(self):
        if self.mat.dim() < 3:
            return self.mat.__class__([1.])
        else:
            return self.mat.__class__(self.mat.shape[0]).fill_(1.)

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        return cls.exp(angle_in_radians)

    def to_angle(self):
        """Recover the rotation angle in rad from the rotation matrix."""
        return self.log()
