import torch

from . import base
from . import utils
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
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = xi.__class__(xi.shape[0], cls.dim, cls.dim).zero_()
        Xi[:, :3, :3] = cls.RotationType.wedge(xi[:, 3:])
        Xi[:, :3, 3] = xi[:, :3]

        return Xi.squeeze_()

    @classmethod
    def vee(cls, Xi):
        if Xi.dim() < 3:
            Xi = Xi.unsqueeze(dim=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = Xi.__class__(Xi.shape[0], cls.dof)
        xi[:, :3] = Xi[:, :3, 3]
        xi[:, 3:] = cls.RotationType.vee(Xi[:, :3, :3])

        return xi.squeeze_()

    @classmethod
    def left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def inv_left_jacobian(cls, xi):
        raise NotImplementedError

    @classmethod
    def exp(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, :3]
        phi = xi[:, 3:]

        rot = cls.RotationType.exp(phi)
        rot_jac = cls.RotationType.left_jacobian(phi)

        if rot_jac.dim() < 3:
            rot_jac.unsqueeze_(dim=0)
        if rho.dim() < 3:
            rho.unsqueeze_(dim=2)

        trans = torch.bmm(rot_jac, rho).squeeze_()

        return cls(rot, trans)

    def log(self):
        phi = self.rot.log()
        inv_rot_jac = self.RotationType.inv_left_jacobian(phi)

        if self.trans.dim() < 2:
            trans = self.trans.unsqueeze(dim=0)
        else:
            trans = self.trans

        if inv_rot_jac.dim() < 3:
            inv_rot_jac.unsqueeze_(dim=0)
        if trans.dim() < 3:
            trans = trans.unsqueeze(dim=2)

        rho = torch.bmm(inv_rot_jac, trans).squeeze_()
        if rho.dim() < 2:
            rho.unsqueeze_(dim=0)
        if phi.dim() < 2:
            phi.unsqueeze_(dim=0)

        return torch.cat([rho, phi], dim=1).squeeze_()

    def adjoint(self):
        rot = self.rot.as_matrix()
        if rot.dim() < 3:
            rot = rot.unsqueeze(dim=0)  # matrix --> batch

        trans = self.trans
        if trans.dim() < 2:
            # vector --> vectorbatch
            trans = trans.unsqueeze(dim=0)

        trans_wedge = self.RotationType.wedge(trans)
        if trans_wedge.dim() < 3:
            trans_wedge.unsqueeze_(dim=0)  # matrix --> batch

        trans_wedge_dot_rot = torch.bmm(trans_wedge, rot)

        zero_block = trans.__class__(rot.shape).zero_()

        return torch.cat([torch.cat([rot, trans_wedge_dot_rot], dim=2),
                          torch.cat([zero_block, rot], dim=2)], dim=1
                         ).squeeze_()

    @classmethod
    def odot(cls, p, directional=False):
        if p.dim() < 2:
            p = p.unsqueeze(dim=0)  # vector --> vectorbatch

        result = p.__class__(p.shape[0], p.shape[1], cls.dof).zero_()

        # Got euclidean coordinates
        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, :3, :3] = torch.eye(3).unsqueeze_(dim=0).expand(
                    p.shape[0], 3, 3)

            result[:, :3, 3:] = cls.RotationType.wedge(-p)

        # Got homogeneous coordinates
        elif p.shape[1] == cls.dim:
            result[:, :3, :3] = \
                p[:, 3].unsqueeze_(dim=1).unsqueeze_(dim=2) * \
                torch.eye(3).unsqueeze_(dim=0).repeat(
                p.shape[0], 1, 1)

            result[:, :3, 3:] = cls.RotationType.wedge(-p[:, :3])

        # Got wrong dimension
        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return result.squeeze_()
