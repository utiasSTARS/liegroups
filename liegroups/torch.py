import torch
import numpy as np

from liegroups import base


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()


def trace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    return (torch.eye(mat.shape[1]) * mat).sum(dim=1).sum(dim=1).squeeze_()


class SpecialOrthogonalBase(base.SpecialOrthogonalBase):
    """Implementation of methods common to SO(N) using PyTorch"""

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid.all() or normalize:
            result = cls(mat)

            if normalize:
                result.normalize(inds=(1 - mat_is_valid).nonzero())

            return result
        else:
            raise ValueError(
                "Invalid rotation matrix. Use normalize=True to handle rounding errors.")

    @classmethod
    def is_valid_matrix(cls, mat):
        if mat.dim() < 3:
            mat = mat.unsqueeze(dim=0)

        if mat.is_cuda:
            shape_check = torch.cuda.ByteTensor(mat.shape[0]).fill_(False)
            det_check = torch.cuda.ByteTensor(mat.shape[0]).fill_(False)
            inv_check = torch.cuda.ByteTensor(mat.shape[0]).fill_(False)
        else:
            shape_check = torch.ByteTensor(mat.shape[0]).fill_(False)
            det_check = torch.ByteTensor(mat.shape[0]).fill_(False)
            inv_check = torch.ByteTensor(mat.shape[0]).fill_(False)

        # Check the shape
        if mat.shape[1:3] != (cls.dim, cls.dim):
            return shape_check
        else:
            shape_check.fill_(True)

        # Determinants of each matrix in the batch should be 1
        det_check = isclose(mat.__class__(
            np.linalg.det(mat.cpu().numpy())), 1.)

        # The transpose of each matrix in the batch should be its inverse
        inv_check = isclose(mat.transpose(2, 1).bmm(mat),
                            torch.eye(cls.dim)).sum(dim=1).sum(dim=1) \
            == cls.dim * cls.dim

        return shape_check & det_check & inv_check

    @classmethod
    def identity(cls, batch_size=1, copy=False):
        mat = torch.eye(cls.dim).expand(batch_size, cls.dim, cls.dim).squeeze()
        return cls(mat)

    def _normalize_one(self, mat):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).
        U, _, V = mat.squeeze().svd()
        S = torch.eye(self.dim)
        if U.is_cuda:
            S = S.cuda()
        S[self.dim - 1, self.dim - 1] = float(np.linalg.det(U.cpu().numpy()) *
                                              np.linalg.det(V.cpu().numpy()))

        mat_normalized = U.mm(S.mm(V.t_()))

        mat.copy_(mat_normalized)
        return mat

    def normalize(self, inds=None):
        if self.mat.dim() < 3:
            self._normalize_one(self.mat)
        else:
            if inds is None:
                inds = range(self.mat.shape[0])

            for batch_ind in inds:
                # Slicing is a copy operation?
                self.mat[batch_ind] = self._normalize_one(self.mat[batch_ind])

    def dot(self, other):
        if isinstance(other, self.__class__):
            # Compound with another rotation
            return self.__class__(torch.matmul(self.mat, other.mat))
        else:
            if other.dim() < 2:
                other = other.unsqueeze(dim=0)  # vector --> matrix
            if other.dim() < 3:
                other = other.unsqueeze(dim=0)  # matrix --> batch

            if self.mat.dim() < 3:
                mat = self.mat.unsqueeze(dim=0).expand(
                    other.shape[0], self.dim, self.dim)  # matrix --> batch
            else:
                mat = self.mat
                if other.shape[0] == 1:
                    other = other.expand(
                        mat.shape[0], other.shape[1], other.shape[2])

            # Transform one or more 2-vectors or fail
            if other.shape[0] != mat.shape[0]:
                raise ValueError("Expected vector-batch batch size of {}, got {}".format(
                    mat.shape[0], other.shape[0]))

            if other.shape[2] == self.dim:
                return torch.bmm(mat, other.transpose(2, 1)).transpose(2, 1).squeeze()
            else:
                raise ValueError(
                    "Vector or vector-batch must have shape ({},), (N,{}), or ({},N,{})".format(self.dim, self.dim, mat.shape[0], self.dim))

    def cuda(self, **kwargs):
        """Return a copy with the underlying tensor on the GPU."""
        return self.__class__(self.mat.cuda(**kwargs))

    def cpu(self):
        """Return a copy with the underlying tensor on the CPU."""
        return self.__class__(self.mat.cpu())


class SO2(SpecialOrthogonalBase):
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
        small_angle_mask = isclose(phi, 0.)
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
        small_angle_mask = isclose(phi, 0.)
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


class SO3(SpecialOrthogonalBase):
    """Rotation matrix in SO(3) using active (alibi) transformations."""
    dim = 3
    dof = 3

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def wedge(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Phi = phi.__class__(phi.shape[0], cls.dim, cls.dim).zero_()
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi.squeeze_()

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi = Phi.unsqueeze(dim=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Phi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        phi = Phi.__class__(Phi.shape[0], cls.dim)
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return phi.squeeze_()

    @classmethod
    def left_jacobian(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()
        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = \
                torch.eye(cls.dim).expand_as(jac[small_angle_inds]) + \
                0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dim)
            s = angle.sin()
            c = angle.cos()

            A = (s / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                torch.eye(cls.dim).unsqueeze_(dim=0).expand_as(
                jac[large_angle_inds])
            B = (1. - s / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                outer(axis, axis)
            C = ((1. - c) / angle).unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds]) * \
                cls.wedge(axis.squeeze())

            jac[large_angle_inds] = A + B + C

        return jac.squeeze_()

    @classmethod
    def inv_left_jacobian(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()
        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = \
                torch.eye(cls.dim).expand_as(jac[small_angle_inds]) - \
                0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dim)

            ha = 0.5 * angle       # half angle
            hacha = ha / ha.tan()  # half angle * cot(half angle)

            ha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])
            hacha.unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(jac[large_angle_inds])

            A = hacha * \
                torch.eye(cls.dim).unsqueeze_(
                    dim=0).expand_as(jac[large_angle_inds])
            B = (1. - hacha) * outer(axis, axis)
            C = -ha * cls.wedge(axis)

            jac[large_angle_inds] = A + B + C

        return jac.squeeze_()

    @classmethod
    def exp(cls, phi):
        if phi.dim() < 2:
            phi = phi.unsqueeze(dim=0)

        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        mat = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        angle = phi.norm(p=2, dim=1)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()
        if len(small_angle_inds) > 0:
            mat[small_angle_inds] = \
                torch.eye(cls.dim).expand_as(mat[small_angle_inds]) + \
                cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            axis = phi[large_angle_inds] / \
                angle.unsqueeze(dim=1).expand(len(angle), cls.dim)
            s = angle.sin().unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(mat[large_angle_inds])
            c = angle.cos().unsqueeze_(dim=1).unsqueeze_(
                dim=2).expand_as(mat[large_angle_inds])

            A = c * torch.eye(cls.dim).unsqueeze_(dim=0).expand_as(
                mat[large_angle_inds])
            B = (1. - c) * outer(axis, axis)
            C = s * cls.wedge(axis)

            mat = A + B + C

        return cls(mat.squeeze_())

    def log(self):
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        phi = mat.__class__(mat.shape[0], self.dof)

        # The cosine of the rotation angle is related to the trace of C
        # Clamp to its proper domain to avoid NaNs from rounding errors
        cos_angle = (0.5 * trace(mat) - 0.5).clamp_(-1., 1.)
        angle = cos_angle.acos()

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(angle, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze_()
        if len(small_angle_inds) > 0:
            phi[small_angle_inds, :] = \
                self.vee(mat[small_angle_inds] -
                         torch.eye(self.dim).expand_as(mat[small_angle_inds]))

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze_()

        if len(large_angle_inds) > 0:
            angle = angle[large_angle_inds]
            sin_angle = angle.sin()

            phi[large_angle_inds, :] = \
                self.vee(
                    (0.5 * angle / sin_angle).unsqueeze_(dim=1).unsqueeze_(dim=1).expand_as(mat) *
                    (mat - mat.transpose(2, 1)))

        return phi.squeeze_()

    def adjoint(self):
        return self.mat

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis."""
        s = angle_in_radians.sin()
        c = angle_in_radians.cos()

        mat = angle_in_radians.__class__(
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

        mat = angle_in_radians.__class__(
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

        mat = angle_in_radians.__class__(
            angle_in_radians.shape[0], cls.dim, cls.dim).zero_()
        mat[:, 2, 2] = 1.
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c

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

    def to_rpy(self):
        """Convert a rotation matrix to RPY Euler angles."""
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        pitch = torch.atan2(-mat[:, 2, 0],
                            torch.sqrt(mat[:, 0, 0]**2 + mat[:, 1, 0]**2))
        yaw = pitch.__class__(pitch.shape)
        roll = pitch.__class__(pitch.shape)

        near_pi_over_two_mask = isclose(pitch, np.pi / 2.)
        near_pi_over_two_inds = near_pi_over_two_mask.nonzero().squeeze_()

        near_neg_pi_over_two_mask = isclose(pitch, -np.pi / 2.)
        near_neg_pi_over_two_inds = near_neg_pi_over_two_mask.nonzero().squeeze_()

        remainder_inds = (1 - (near_pi_over_two_mask |
                               near_neg_pi_over_two_mask)).nonzero().squeeze_()

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
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if quat.dim() < 2:
            quat = quat.unsqueeze(dim=0)

        if not allclose(quat.norm(p=2, dim=1), 1.):
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
        mat = quat.__class__(quat.shape[0], cls.dim, cls.dim)

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

    def to_quaternion(self, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if self.mat.dim() < 3:
            R = self.mat.unsqueeze(dim=0)
        else:
            R = self.mat

        qw = 0.5 * torch.sqrt(1. + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
        qx = qw.__class__(qw.shape)
        qy = qw.__class__(qw.shape)
        qz = qw.__class__(qw.shape)

        near_zero_mask = isclose(qw, 0.)

        if sum(near_zero_mask) > 0:
            cond1_mask = near_zero_mask & \
            (R[:, 0, 0] > R[:, 1, 1]).squeeze_() & \
            (R[:, 0, 0] > R[:, 2, 2]).squeeze_()
            cond1_inds = cond1_mask.nonzero().squeeze_()

            if len(cond1_inds) > 0:
                R_cond1 = R[cond1_inds]
                d = 2. * np.sqrt(1. + R_cond1[:,0, 0] - 
                R_cond1[:,1, 1] - R_cond1[:,2, 2])
                qw[cond1_inds] = (R_cond1[:,2, 1] - R_cond1[:,1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:,1, 0] + R_cond1[:,0, 1]) / d
                qz[cond1_inds] = (R_cond1[:,0, 2] + R_cond1[:,2, 0]) / d

            cond2_mask = near_zero_mask & (R[:, 1, 1] > R[:, 2, 2]).squeeze_()
            cond2_inds = cond2_mask.nonzero().squeeze_()
            
            if len(cond2_inds) > 0:
                R_cond2 = R[cond2_inds]
                d = 2. * np.sqrt(1. + R_cond2[:,1, 1] - 
                R_cond2[:,0, 0] - R_cond2[:,2, 2])
                qw[cond2_inds] = (R_cond2[:,0, 2] - R_cond2[:,2, 0]) / d
                qx[cond2_inds] = (R_cond2[:,1, 0] + R_cond2[:,0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:,2, 1] + R_cond2[:,1, 2]) / d
            
            cond3_mask = near_zero_mask & (1-cond1_mask) & (1-cond2_mask)
            cond3_inds = cond3_mask.nonzero().squeeze_()

            if len(cond3_inds) > 0:
                R_cond3 = R[cond3_inds]
                d = 2. * np.sqrt(1. + R_cond3[:,2, 2] - R_cond3[:,0, 0] - R_cond3[:,1, 1])
                qw[cond3_inds] = (R_cond3[:,1, 0] - R_cond3[:,0, 1]) / d
                qx[cond3_inds] = (R_cond3[:,0, 2] + R_cond3[:,2, 0]) / d
                qy[cond3_inds] = (R_cond3[:,2, 1] + R_cond3[:,1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = 1 - near_zero_mask
        far_zero_inds = far_zero_mask.nonzero().squeeze_()
        if len(far_zero_inds) > 0:
            R_fz = R[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:,2, 1] - R_fz[:,1, 2]) / d
            qy[far_zero_inds] = (R_fz[:,0, 2] - R_fz[:,2, 0]) / d
            qz[far_zero_inds] = (R_fz[:,2, 1] - R_fz[:,1, 2]) / d

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


class SpecialEuclideanlBase(base.SpecialEuclideanBase):
    """Implementation of methods common to SE(N) using PyTorch"""

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    def cuda(self, **kwargs):
        """Return a copy with the underlying tensors on the GPU."""
        return self.__class__(self.rot.cuda(**kwargs), self.trans.cuda(**kwargs))

    def cpu(self):
        """Return a copy with the underlying tensors on the CPU."""
        return self.__class__(self.rot.cpu(), self.trans.cpu())


class SE2(SpecialEuclideanlBase):
    pass


class SE3(SpecialEuclideanlBase):
    pass
