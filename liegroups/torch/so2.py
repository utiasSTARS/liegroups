import torch

from . import _base
from . import utils


class SO2Matrix(_base.SOMatrixBase):
    """See :mod:`liegroups.SO2`"""
    dim = 2
    dof = 1

    def adjoint(self):
        if self.mat.dim() < 3:
            return self.mat.__class__([1.])
        else:
            return self.mat.__class__(self.mat.shape[0]).fill_(1.)

    @classmethod
    def exp(cls, phi):
        if phi.dim() < 1:
            phi = phi.unsqueeze(dim=0)

        s = phi.sin()
        c = phi.cos()

        mat = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c

        return cls(mat.squeeze_())

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        return cls.exp(angle_in_radians)

    @classmethod
    def inv_left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        if phi.dim() < 1:
            phi = phi.unsqueeze(dim=0)

        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                - 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = phi[large_angle_inds]
            ha = 0.5 * angle       # half angle
            hacha = ha / ha.tan()  # half angle * cot(half angle)

            ha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])
            hacha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])

            A = hacha * \
                torch.eye(cls.dim).unsqueeze_(
                    dim=0).expand_as(jac[large_angle_inds])
            B = -ha * cls.wedge(phi.__class__([1.]))

            jac[large_angle_inds] = A + B

        return jac.squeeze_()

    @classmethod
    def left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        if phi.dim() < 1:
            phi = phi.unsqueeze(dim=0)

        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                + 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = phi[large_angle_inds]
            s = angle.sin()
            c = angle.cos()

            A = (s / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                torch.eye(cls.dim).unsqueeze_(dim=0).expand_as(
                jac[large_angle_inds])
            B = ((1. - c) / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                cls.wedge(phi.__class__([1.]))

            jac[large_angle_inds] = A + B

        return jac.squeeze_()

    def log(self):
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        s = mat[:, 1, 0]
        c = mat[:, 0, 0]

        return torch.atan2(s, c).squeeze_()

    def to_angle(self):
        """Recover the rotation angle in rad from the rotation matrix."""
        return self.log()

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi = Phi.unsqueeze(dim=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(cls.dim, cls.dim, cls.dim, cls.dim))

        return Phi[:, 1, 0].squeeze_()

    @classmethod
    def wedge(cls, phi):
        if phi.dim() < 1:
            phi = phi.unsqueeze(dim=0)  # scalar --> vector
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=1)  # vector --> matrix (N --> Nx1)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Phi = phi.new_zeros(phi.shape[0], cls.dim, cls.dim)
        Phi[:, 0, 1] = -phi[:, 0]
        Phi[:, 1, 0] = phi[:, 0]
        return Phi.squeeze_()
