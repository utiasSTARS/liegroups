import torch

from . import _base
from . import utils
from .so2 import SO2Matrix


class SE2Matrix(_base.SEMatrixBase):
    """See :mod:`liegroups.SE2`"""
    dim = 3
    dof = 3
    RotationType = SO2Matrix

    def adjoint(self):
        rot_part = self.rot.as_matrix()
        if rot_part.dim() < 3:
            rot_part = rot_part.unsqueeze(dim=0)  # matrix --> batch

        trans = self.trans
        if trans.dim() < 2:
            # vector --> vectorbatch
            trans = trans.unsqueeze(dim=0)

        trans_part = trans.new_empty(
            trans.shape[0], trans.shape[1], 1)
        trans_part[:, 0, 0] = trans[:, 1]
        trans_part[:, 1, 0] = -trans[:, 0]

        bottom_row = trans.new_zeros(self.dof)
        bottom_row[-1] = 1.
        bottom_row = bottom_row.unsqueeze_(dim=0).unsqueeze_(
            dim=0).expand(trans.shape[0], 1, self.dof)

        return torch.cat([torch.cat([rot_part, trans_part], dim=2),
                          bottom_row], dim=1).squeeze_()

    @classmethod
    def exp(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, 0:2]
        phi = xi[:, 2]

        rot = cls.RotationType.exp(phi)
        rot_jac = cls.RotationType.left_jacobian(phi)

        if rot_jac.dim() < 3:
            rot_jac.unsqueeze_(dim=0)
        if rho.dim() < 3:
            rho.unsqueeze_(dim=2)

        trans = torch.bmm(rot_jac, rho).squeeze_()

        return cls(rot, trans)

    @classmethod
    def inv_left_jacobian(cls, xi):

        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, 0:2]  # translation part
        phi = xi[:, 2]  # rotation part

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        phi_sq = phi * phi

        small_angle_mask = utils.isclose(phi_sq, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_(dim=1)

        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero().squeeze_(dim=1)

        jac = torch.zeros((xi.shape[0], cls.dof, cls.dof)).to(xi.device)

        jac[small_angle_inds, 0, 0] = -(96*(phi_sq[small_angle_inds] - 6))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)
        jac[small_angle_inds, 0, 1] = -(24*phi[small_angle_inds]*(phi_sq[small_angle_inds] - 12))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)
        jac[small_angle_inds, 0, 2] = (4*(12*phi[small_angle_inds]*rho[small_angle_inds,0] - 72*rho[small_angle_inds,1] + 12*phi_sq[small_angle_inds]*rho[small_angle_inds,1] - 12*phi_sq[small_angle_inds]*rho[small_angle_inds,1] + phi_sq[small_angle_inds]*phi_sq[small_angle_inds]*rho[small_angle_inds,1] + phi_sq[small_angle_inds]*phi[small_angle_inds]*rho[small_angle_inds,0]))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)
        jac[small_angle_inds, 1, 0] = (24*phi[small_angle_inds]*(phi_sq[small_angle_inds] - 12))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)
        jac[small_angle_inds, 1, 1] = -(96*(phi_sq[small_angle_inds] - 6))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)
        jac[small_angle_inds, 1, 2] = (4*(72*rho[small_angle_inds,0] - 12*phi_sq[small_angle_inds]*rho[small_angle_inds,0] + 12*phi[small_angle_inds]*rho[small_angle_inds,1] + 12*phi_sq[small_angle_inds]*rho[small_angle_inds,0] - phi_sq[small_angle_inds]*phi_sq[small_angle_inds]*rho[small_angle_inds,0] + phi_sq[small_angle_inds]*phi[small_angle_inds]*rho[small_angle_inds,1]))/(phi_sq[small_angle_inds]**2*phi_sq[small_angle_inds] + 16*phi_sq[small_angle_inds]**2 - 24*phi_sq[small_angle_inds]*phi_sq[small_angle_inds] - 192*phi_sq[small_angle_inds] + 144*phi_sq[small_angle_inds] + 576)

        jac[large_angle_inds, 0, 0] = (sin_phi[large_angle_inds]*phi[large_angle_inds])/(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1)
        jac[large_angle_inds, 0, 1] = -(phi[large_angle_inds]*(cos_phi[large_angle_inds] - 1))/(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1)
        jac[large_angle_inds, 0, 2] = (phi[large_angle_inds]*(rho[large_angle_inds,0] - 2*cos_phi[large_angle_inds]*rho[large_angle_inds,0] - phi[large_angle_inds]*rho[large_angle_inds,1] + cos_phi[large_angle_inds]**2*rho[large_angle_inds,0] + sin_phi[large_angle_inds]**2*rho[large_angle_inds,0] + cos_phi[large_angle_inds]*phi[large_angle_inds]*rho[large_angle_inds,1] - sin_phi[large_angle_inds]*phi[large_angle_inds]*rho[large_angle_inds,0]))/(phi_sq[large_angle_inds]*(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1))
        jac[large_angle_inds, 1, 0] = (phi[large_angle_inds]*(cos_phi[large_angle_inds] - 1))/(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1)
        jac[large_angle_inds, 1, 1] = (sin_phi[large_angle_inds]*phi[large_angle_inds])/(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1)
        jac[large_angle_inds, 1, 2] = (phi[large_angle_inds]*(rho[large_angle_inds,1] - 2*cos_phi[large_angle_inds]*rho[large_angle_inds,1] + phi[large_angle_inds]*rho[large_angle_inds,0] + cos_phi[large_angle_inds]**2*rho[large_angle_inds,1] + sin_phi[large_angle_inds]**2*rho[large_angle_inds,1] - cos_phi[large_angle_inds]*phi[large_angle_inds]*rho[large_angle_inds,0] - sin_phi[large_angle_inds]*phi[large_angle_inds]*rho[large_angle_inds,1]))/(phi_sq[large_angle_inds]*(cos_phi[large_angle_inds]**2 - 2*cos_phi[large_angle_inds] + sin_phi[large_angle_inds]**2 + 1))
        
        jac[:, 2, 0] = 0
        jac[:, 2, 1] = 0
        jac[:, 2, 2] = 1

        return jac.squeeze_()

    @classmethod
    def left_jacobian(cls, xi):

        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, 0:2]  # translation part
        phi = xi[:, 2]  # rotation part

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        phi_sq = phi * phi

        small_angle_mask = utils.isclose(phi_sq, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_(dim=1)

        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero().squeeze_(dim=1)

        jac = torch.zeros((xi.shape[0], cls.dof, cls.dof)).to(xi.device)

        jac[small_angle_inds, 0, 0] = 1 - 1./6. * phi_sq[small_angle_inds]
        jac[small_angle_inds, 0, 1] = -(0.5 * phi[small_angle_inds] - 1./24. * phi[small_angle_inds] * phi_sq[small_angle_inds])
        jac[small_angle_inds, 0, 2] = rho[small_angle_inds,1] / 2. + phi[small_angle_inds] * rho[small_angle_inds,0] / 6.
        jac[small_angle_inds, 1, 0] = 0.5 * phi[small_angle_inds] - 1./24. * phi[small_angle_inds] * phi_sq[small_angle_inds]
        jac[small_angle_inds, 1, 1] = 1 - 1./6. * phi_sq[small_angle_inds]
        jac[small_angle_inds, 1, 2] = -rho[small_angle_inds,0] / 2. + phi[small_angle_inds] * rho[small_angle_inds,1] / 6.

        jac[large_angle_inds, 0, 0] = sin_phi[large_angle_inds] / phi[large_angle_inds]
        jac[large_angle_inds, 0, 1] = -(1 - cos_phi[large_angle_inds]) / phi[large_angle_inds]
        jac[large_angle_inds, 0, 2] = ( rho[large_angle_inds,1] + phi[large_angle_inds]*rho[large_angle_inds,0] - rho[large_angle_inds,1]*cos_phi[large_angle_inds] - rho[large_angle_inds,0]*sin_phi[large_angle_inds])/phi_sq[large_angle_inds]
        jac[large_angle_inds, 1, 0] = (1 - cos_phi[large_angle_inds]) / phi[large_angle_inds]
        jac[large_angle_inds, 1, 1] = sin_phi[large_angle_inds] / phi[large_angle_inds]
        jac[large_angle_inds, 1, 2] = (-rho[large_angle_inds,0] + phi[large_angle_inds]*rho[large_angle_inds,1] + rho[large_angle_inds,0]*cos_phi[large_angle_inds] - rho[large_angle_inds,1]*sin_phi[large_angle_inds])/phi_sq[large_angle_inds]

        jac[:, 2, 0] = 0
        jac[:, 2, 1] = 0
        jac[:, 2, 2] = 1

        return jac.squeeze_()

    def log(self):
        phi = self.rot.log()
        inv_rot_jac = self.RotationType.inv_left_jacobian(phi)

        if self.trans.dim() < 2:
            trans = self.trans.unsqueeze(dim=0)
        else:
            trans = self.trans

        if phi.dim() < 1:
            phi.unsqueeze_(dim=0)
        phi.unsqueeze_(dim=1)  # because phi is 1-dimensional for SE2

        if inv_rot_jac.dim() < 3:
            inv_rot_jac.unsqueeze_(dim=0)
        if trans.dim() < 3:
            trans = trans.unsqueeze(dim=2)

        rho = torch.bmm(inv_rot_jac, trans).squeeze_()
        if rho.dim() < 2:
            rho.unsqueeze_(dim=0)

        return torch.cat([rho, phi], dim=1).squeeze_()

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
                result[:, 0:2, 0:2] = torch.eye(
                    cls.RotationType.dim).unsqueeze_(dim=0).expand(
                        p.shape[0], cls.RotationType.dim, cls.RotationType.dim)

            result[:, 0:2, 2] = torch.mm(
                cls.RotationType.wedge(p.__class__([1.])),
                p.transpose(1, 0)).transpose_(1, 0)

        # Got homogeneous coordinates
        elif p.shape[1] == cls.dim:
            result[:, 0:2, 0:2] = \
                p[:, 2].unsqueeze_(dim=1).unsqueeze_(dim=2) * \
                torch.eye(
                cls.RotationType.dim).unsqueeze_(dim=0).repeat(
                p.shape[0], 1, 1)

            result[:, 0:2, 2] = torch.mm(
                cls.RotationType.wedge(p.__class__([1.])),
                p[:, 0:2].transpose_(1, 0)).transpose_(1, 0)

        # Got wrong dimension
        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return result.squeeze_()

    @classmethod
    def vee(cls, Xi):
        if Xi.dim() < 3:
            Xi = Xi.unsqueeze(dim=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = Xi.new_empty(Xi.shape[0], cls.dof)
        xi[:, 0:2] = Xi[:, 0:2, 2]
        xi[:, 2] = cls.RotationType.vee(Xi[:, 0:2, 0:2])

        return xi.squeeze_()

    @classmethod
    def wedge(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = xi.new_zeros(xi.shape[0], cls.dim, cls.dim)
        Xi[:, 0:2, 0:2] = cls.RotationType.wedge(xi[:, 2])
        Xi[:, 0:2, 2] = xi[:, 0:2]

        return Xi.squeeze_()
