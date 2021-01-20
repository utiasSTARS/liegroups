import torch
import numpy as np

from . import _base
from . import utils


class SO3Matrix(_base.SOMatrixBase):
    """See :mod:`liegroups.SO3`"""
    dim = 3
    dof = 3

    def adjoint(self):
        return self.mat

    @classmethod
    def exp(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        mat = phi.new_empty(phi.shape[0], cls.dim, cls.dim)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(small_angle_inds) > 0:
            mat[small_angle_inds] = \
                torch.eye(cls.dim, dtype=phi.dtype).expand_as(mat[small_angle_inds]) + \
                cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dim)
            s = angle.sin().unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(mat[large_angle_inds])
            c = angle.cos().unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(mat[large_angle_inds])

            A = c * torch.eye(cls.dim, dtype=phi.dtype).unsqueeze_(dim=0).expand_as(
                mat[large_angle_inds])
            B = (1. - c) * utils.outer(axis, axis)
            C = s * cls.wedge(axis)

            mat[large_angle_inds] = A + B + C

        return cls(mat.squeeze_())

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if quat.dim() < 2:
            quat = quat.unsqueeze(dim=0)

        if not utils.allclose(quat.norm(p=2, dim=1), 1.):
            raise ValueError("Quaternions must be unit length")

        if ordering is 'xyzw':
            qx = quat[:, 0]
            qy = quat[:, 1]
            qz = quat[:, 2]
            qw = quat[:, 3]
        elif ordering is 'wxyz':
            qw = quat[:, 0]
            qx = quat[:, 1]
            qy = quat[:, 2]
            qz = quat[:, 3]
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        # Form the matrix
        mat = quat.new_empty(quat.shape[0], cls.dim, cls.dim)

        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
        mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
        mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)

        return cls(mat.squeeze_())

    @classmethod
    def from_rpy(cls, rpy):
        """Form a rotation matrix from RPY Euler angles."""
        if rpy.dim() < 2:
            rpy = rpy.unsqueeze(dim=0)

        roll = rpy[:, 0]
        pitch = rpy[:, 1]
        yaw = rpy[:, 2]
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

    @classmethod
    def inv_left_jacobian(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        jac = phi.new_empty(phi.shape[0], cls.dof, cls.dof)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = \
                torch.eye(cls.dof, dtype=phi.dtype).expand_as(jac[small_angle_inds]) - \
                0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dof)

            ha = 0.5 * angle       # half angle
            hacha = ha / ha.tan()  # half angle * cot(half angle)

            ha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])
            hacha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])

            A = hacha * \
                torch.eye(cls.dof, dtype=phi.dtype).unsqueeze_(
                    dim=0).expand_as(jac[large_angle_inds])
            B = (1. - hacha) * utils.outer(axis, axis)
            C = -ha * cls.wedge(axis)

            jac[large_angle_inds] = A + B + C

        return jac.squeeze_()

    @classmethod
    def left_jacobian(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        jac = phi.new_empty(phi.shape[0], cls.dof, cls.dof)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = \
                torch.eye(cls.dof, dtype=phi.dtype).expand_as(jac[small_angle_inds]) + \
                0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dof)
            s = angle.sin()
            c = angle.cos()

            A = (s / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                torch.eye(cls.dof, dtype=phi.dtype).unsqueeze_(dim=0).expand_as(
                jac[large_angle_inds])
            B = (1. - s / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                utils.outer(axis, axis)
            C = ((1. - c) / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                cls.wedge(axis.squeeze())

            jac[large_angle_inds] = A + B + C

        return jac.squeeze_()

    def log(self):
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        phi = mat.new_empty(mat.shape[0], self.dof)

        # The cosine of the rotation angle is related to the utils.trace of C
        # Clamp to its proper domain to avoid NaNs from rounding errors
        cos_angle = (0.5 * utils.trace(mat) - 0.5).clamp_(-1., 1.)
        angle = cos_angle.acos()

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = utils.isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(small_angle_inds) > 0:
            phi[small_angle_inds, :] = \
                self.vee(mat[small_angle_inds] -
                         torch.eye(self.dim, dtype=mat.dtype).expand_as(mat[small_angle_inds]))

        # Otherwise...
        large_angle_mask = small_angle_mask.logical_not()
        large_angle_inds = large_angle_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            sin_angle = angle.sin()
            phi[large_angle_inds, :] = \
                self.vee(
                    (0.5 * angle / sin_angle).unsqueeze_(dim=1).unsqueeze_(dim=1).expand_as(mat[large_angle_inds]) *
                    (mat[large_angle_inds] - mat[large_angle_inds].transpose(2, 1)))

        return phi.squeeze_()

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis."""
        s = angle_in_radians.sin()
        c = angle_in_radians.cos()

        mat = angle_in_radians.new_empty(
            angle_in_radians.shape[0], cls.dim, cls.dim).zero_()
        mat[:, 0, 0] = 1.
        mat[:, 1, 1] = c
        mat[:, 1, 2] = -s
        mat[:, 2, 1] = s
        mat[:, 2, 2] = c

        return cls(mat.squeeze_())

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis."""
        s = angle_in_radians.sin()
        c = angle_in_radians.cos()

        mat = angle_in_radians.new_empty(
            angle_in_radians.shape[0], cls.dim, cls.dim).zero_()
        mat[:, 1, 1] = 1.
        mat[:, 0, 0] = c
        mat[:, 0, 2] = s
        mat[:, 2, 0] = -s
        mat[:, 2, 2] = c

        return cls(mat.squeeze_())

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis."""
        s = angle_in_radians.sin()
        c = angle_in_radians.cos()

        mat = angle_in_radians.new_empty(
            angle_in_radians.shape[0], cls.dim, cls.dim).zero_()
        mat[:, 2, 2] = 1.
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c

        return cls(mat.squeeze_())

    def to_quaternion(self, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if self.mat.dim() < 3:
            R = self.mat.unsqueeze(dim=0)
        else:
            R = self.mat

        qw = 0.5 * torch.sqrt(1. + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
        qx = qw.new_empty(qw.shape)
        qy = qw.new_empty(qw.shape)
        qz = qw.new_empty(qw.shape)

        near_zero_mask = utils.isclose(qw, 0.)

        if sum(near_zero_mask) > 0:
            cond1_mask = near_zero_mask & \
                (R[:, 0, 0] > R[:, 1, 1]).squeeze_() & \
                (R[:, 0, 0] > R[:, 2, 2]).squeeze_()
            cond1_inds = cond1_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond1_inds) > 0:
                R_cond1 = R[cond1_inds]
                d = 2. * np.sqrt(1. + R_cond1[:, 0, 0] -
                                 R_cond1[:, 1, 1] - R_cond1[:, 2, 2])
                qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

            cond2_mask = near_zero_mask & (R[:, 1, 1] > R[:, 2, 2]).squeeze_()
            cond2_inds = cond2_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond2_inds) > 0:
                R_cond2 = R[cond2_inds]
                d = 2. * np.sqrt(1. + R_cond2[:, 1, 1] -
                                 R_cond2[:, 0, 0] - R_cond2[:, 2, 2])
                qw[cond2_inds] = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

            cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
            cond3_inds = cond3_mask.nonzero(as_tuple=False).squeeze_(dim=1)

            if len(cond3_inds) > 0:
                R_cond3 = R[cond3_inds]
                d = 2. * \
                    np.sqrt(1. + R_cond3[:, 2, 2] -
                            R_cond3[:, 0, 0] - R_cond3[:, 1, 1])
                qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = near_zero_mask.logical_not()
        far_zero_inds = far_zero_mask.nonzero(as_tuple=False).squeeze_(dim=1)
        if len(far_zero_inds) > 0:
            R_fz = R[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
            qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
            qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

        # Check ordering last
        if ordering is 'xyzw':
            quat = torch.cat([qx.unsqueeze_(dim=1),
                              qy.unsqueeze_(dim=1),
                              qz.unsqueeze_(dim=1),
                              qw.unsqueeze_(dim=1)], dim=1).squeeze_()
        elif ordering is 'wxyz':
            quat = torch.cat([qw.unsqueeze_(dim=1),
                              qx.unsqueeze_(dim=1),
                              qy.unsqueeze_(dim=1),
                              qz.unsqueeze_(dim=1)], dim=1).squeeze_()
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        return quat

    def to_rpy(self):
        """Convert a rotation matrix to RPY Euler angles."""
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        pitch = torch.atan2(-mat[:, 2, 0],
                            torch.sqrt(mat[:, 0, 0]**2 + mat[:, 1, 0]**2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = utils.isclose(pitch, np.pi / 2.)
        near_pi_over_two_inds = near_pi_over_two_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        near_neg_pi_over_two_mask = utils.isclose(pitch, -np.pi / 2.)
        near_neg_pi_over_two_inds = near_neg_pi_over_two_mask.nonzero(as_tuple=False).squeeze_(dim=1)

        remainder_inds = (near_pi_over_two_mask |
                          near_neg_pi_over_two_mask).logical_not().nonzero(as_tuple=False).squeeze_(dim=1)

        if len(near_pi_over_two_inds) > 0:
            yaw[near_pi_over_two_inds] = 0.
            roll[near_pi_over_two_inds] = torch.atan2(
                mat[near_pi_over_two_inds, 0, 1],
                mat[near_pi_over_two_inds, 1, 1])

        if len(near_neg_pi_over_two_inds) > 0:
            yaw[near_pi_over_two_inds] = 0.
            roll[near_pi_over_two_inds] = -torch.atan2(
                mat[near_pi_over_two_inds, 0, 1],
                mat[near_pi_over_two_inds, 1, 1])

        if len(remainder_inds) > 0:
            sec_pitch = 1. / pitch[remainder_inds].cos()
            remainder_mats = mat[remainder_inds]
            yaw = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                              remainder_mats[:, 0, 0] * sec_pitch)
            roll = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                               remainder_mats[:, 2, 2] * sec_pitch)

        return torch.cat([roll.unsqueeze_(dim=1),
                          pitch.unsqueeze_(dim=1),
                          yaw.unsqueeze_(dim=1)], dim=1).squeeze_()

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi = Phi.unsqueeze(dim=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Phi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        phi = Phi.new_empty(Phi.shape[0], cls.dim)
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return phi.squeeze_()

    @classmethod
    def wedge(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Phi = phi.new_empty(phi.shape[0], cls.dim, cls.dim).zero_()
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi.squeeze_()


class SO3Quaternion(_base.VectorLieGroupBase):
    pass
