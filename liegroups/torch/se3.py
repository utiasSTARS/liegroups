import torch

from . import _base
from . import utils
from .so3 import SO3Matrix


class SE3Matrix(_base.SEMatrixBase):
    """See :mod:`liegroups.SE3` """
    dim = 4
    dof = 6
    RotationType = SO3Matrix

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

        trans_wedge_bmm_rot = torch.bmm(trans_wedge, rot)

        zero_block = trans.new_empty(rot.shape).zero_()

        return torch.cat([torch.cat([rot, trans_wedge_bmm_rot], dim=2),
                          torch.cat([zero_block, rot], dim=2)], dim=1
                         ).squeeze_()

    @classmethod
    def curlyvee(cls, Psi):
        if Psi.dim() < 3:
            Psi = Psi.unsqueeze(dim=0)

        if Psi.shape[1:] != (cls.dof, cls.dof):
            raise ValueError("Psi must have shape ({},{}) or (N,{},{})".format(
                cls.dof, cls.dof, cls.dof, cls.dof))

        xi = Psi.new_empty(Psi.shape[0], cls.dof)
        xi[:, :3] = cls.RotationType.vee(Psi[:, :3, 3:])
        xi[:, 3:] = cls.RotationType.vee(Psi[:, :3, :3])

        return xi.squeeze_()

    @classmethod
    def curlywedge(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Psi = xi.new_empty(xi.shape[0], cls.dof, cls.dof).zero_()
        Psi[:, :3, :3] = cls.RotationType.wedge(xi[:, 3:])
        Psi[:, :3, 3:] = cls.RotationType.wedge(xi[:, :3])
        Psi[:, 3:, 3:] = Psi[:, :3, :3]

        return Psi.squeeze_()

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

    @classmethod
    def left_jacobian_Q_matrix(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, :3]  # translation part
        phi = xi[:, 3:]  # rotation part

        rx = cls.RotationType.wedge(rho)
        if rx.dim() < 3:
            rx.unsqueeze_(dim=0)

        px = cls.RotationType.wedge(phi)
        if px.dim() < 3:
            px.unsqueeze_(dim=0)

        ph = phi.norm(p=2, dim=1)
        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = ph.cos()
        sph = ph.sin()

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        m2 = m2.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m3 = m3.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)
        m4 = m4.unsqueeze_(dim=1).unsqueeze_(dim=2).expand_as(rx)

        t1 = rx
        t2 = px.bmm(rx) + rx.bmm(px) + px.bmm(rx).bmm(px)
        t3 = px.bmm(px).bmm(rx) + rx.bmm(px).bmm(px) - 3. * px.bmm(rx).bmm(px)
        t4 = px.bmm(rx).bmm(px).bmm(px) + px.bmm(px).bmm(rx).bmm(px)

        Q = m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

        return Q.squeeze_()

    @classmethod
    def inv_left_jacobian(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, :3]  # translation part
        phi = xi[:, 3:]  # rotation part

        jac = phi.new_empty(phi.shape[0], cls.dof, cls.dof)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(small_angle_inds) > 0:

            # Create an identity matrix with a tensor type that matches the input
            I = phi.new_empty(cls.dof, cls.dof)
            torch.eye(cls.dof, out=I)

            jac[small_angle_inds] = \
                I.expand_as(jac[small_angle_inds]) - \
                0.5 * cls.curlywedge(xi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            so3_inv_jac = cls.RotationType.inv_left_jacobian(
                phi[large_angle_inds])
            if so3_inv_jac.dim() < 3:
                so3_inv_jac.unsqueeze_(dim=0)

            Q_mat = cls.left_jacobian_Q_matrix(xi[large_angle_inds])
            if Q_mat.dim() < 3:
                Q_mat.unsqueeze_(dim=0)

            zero_block = phi.new_empty(Q_mat.shape).zero_()
            inv_jac_Q_inv_jac = so3_inv_jac.bmm(Q_mat).bmm(so3_inv_jac)

            jac[large_angle_inds] = torch.cat(
                [torch.cat([so3_inv_jac, -inv_jac_Q_inv_jac], dim=2),
                 torch.cat([zero_block, so3_inv_jac], dim=2)], dim=1)

        return jac.squeeze_()

    @classmethod
    def left_jacobian(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        rho = xi[:, :3]  # translation part
        phi = xi[:, 3:]  # rotation part

        jac = phi.new_empty(phi.shape[0], cls.dof, cls.dof)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(small_angle_inds) > 0:
            # Create an identity matrix with a tensor type that matches the input
            I = phi.new_empty(cls.dof, cls.dof)
            torch.eye(cls.dof, out=I)

            jac[small_angle_inds] = \
                I.expand_as(jac[small_angle_inds]) + \
                0.5 * cls.curlywedge(xi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            so3_jac = cls.RotationType.left_jacobian(phi[large_angle_inds])
            if so3_jac.dim() < 3:
                so3_jac.unsqueeze_(dim=0)

            Q_mat = cls.left_jacobian_Q_matrix(xi[large_angle_inds])
            if Q_mat.dim() < 3:
                Q_mat.unsqueeze_(dim=0)

            zero_block = phi.new_empty(Q_mat.shape).zero_()

            jac[large_angle_inds] = torch.cat(
                [torch.cat([so3_jac, Q_mat], dim=2),
                 torch.cat([zero_block, so3_jac], dim=2)], dim=1)

        return jac.squeeze_()

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

    @classmethod
    def odot(cls, p, directional=False):
        if p.dim() < 2:
            p = p.unsqueeze(dim=0)  # vector --> vectorbatch

        result = p.new_empty(p.shape[0], p.shape[1], cls.dof).zero_()

        # Got euclidean coordinates
        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, :3, :3] = torch.eye(3, dtype=p.dtype).unsqueeze_(dim=0).expand(
                    p.shape[0], 3, 3)

            result[:, :3, 3:] = cls.RotationType.wedge(-p)

        # Got homogeneous coordinates
        elif p.shape[1] == cls.dim:
            result[:, :3, :3] = \
                p[:, 3].unsqueeze_(dim=1).unsqueeze_(dim=2) * \
                torch.eye(3, dtype=p.dtype).unsqueeze_(dim=0).repeat(
                p.shape[0], 1, 1)

            result[:, :3, 3:] = cls.RotationType.wedge(-p[:, :3])

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
        xi[:, :3] = Xi[:, :3, 3]
        xi[:, 3:] = cls.RotationType.vee(Xi[:, :3, :3])

        return xi.squeeze_()

    @classmethod
    def wedge(cls, xi):
        if xi.dim() < 2:
            xi = xi.unsqueeze(dim=0)

        if xi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = xi.new_empty(xi.shape[0], cls.dim, cls.dim).zero_()
        Xi[:, :3, :3] = cls.RotationType.wedge(xi[:, 3:])
        Xi[:, :3, 3] = xi[:, :3]

        return Xi.squeeze_()
