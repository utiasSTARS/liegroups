import torch
import numpy as np

from . import base


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


class SO2(base.SpecialOrthogonalGroup):
    """Rotation matrix in SO(2) using active (alibi) transformations."""
    dim = 2
    dof = 1

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
        mat = torch.eye(cls.dim)
        if batch_size > 1:
            if copy:
                mat = mat.repeat(batch_size, 1, 1)
            else:
                mat = mat.expand(batch_size, cls.dim, cls.dim)

        return cls(mat)

    @classmethod
    def wedge(cls, phi):
        Phi = phi.__class__(phi.shape[0], cls.dim, cls.dim).zero_()
        Phi[:, 0, 1] = -phi
        Phi[:, 1, 0] = phi
        return Phi.squeeze()

    @classmethod
    def vee(cls, Phi):
        if Phi.dim() < 3:
            Phi.unsqueeze_(dim=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(cls.dim, cls.dim, cls.dim, cls.dim))

        return Phi[:, 1, 0].squeeze()

    @classmethod
    def left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze()

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                + 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze()

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

        return jac.squeeze()

    @classmethod
    def inv_left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        jac = phi.__class__(phi.shape[0], cls.dim, cls.dim)

        # Near phi==0, use first order Taylor expansion
        small_angle_mask = isclose(phi, 0.)
        small_angle_inds = small_angle_mask.nonzero().squeeze()

        if len(small_angle_inds) > 0:
            jac[small_angle_inds] = torch.eye(cls.dim).expand(
                len(small_angle_inds), cls.dim, cls.dim) \
                - 0.5 * cls.wedge(phi[small_angle_inds])

        # Otherwise...
        large_angle_mask = 1 - small_angle_mask  # element-wise not
        large_angle_inds = large_angle_mask.nonzero().squeeze()

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

        return jac.squeeze()

    @classmethod
    def exp(cls, phi):
        s = phi.sin()
        c = phi.cos()

        mat = phi.__class__(phi.shape[0], cls.dim, cls.dim)
        mat[:, 0, 0] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        mat[:, 1, 1] = c

        return cls(mat.squeeze())

    def log(self):
        if self.mat.dim() < 3:
            mat = self.mat.unsqueeze(dim=0)
        else:
            mat = self.mat

        s = mat[:, 1, 0]
        c = mat[:, 0, 0]

        return torch.atan2(s, c).squeeze()

    def inv(self):
        if self.mat.dim() < 3:
            return self.__class__(self.mat.transpose(1, 0))
        else:
            return self.__class__(self.mat.transpose(2, 1))

    def adjoint(self):
        if self.mat.dim() < 3:
            return self.mat.__class__([1.])
        else:
            return self.mat.__class__(self.mat.shape[0]).fill_(1.)

    def _normalize_one(self, mat):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).
        U, _, V = mat.squeeze().svd()
        S = torch.eye(self.dim)
        if U.is_cuda:
            S = S.cuda()
        S[1, 1] = float(np.linalg.det(U.cpu().numpy()) *
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
                other = other.unsqueeze(dim=1)

            # Transform one or more 2-vectors or fail
            if other.shape[0] == self.dim:
                return torch.matmul(self.mat, other).squeeze()
            else:
                raise ValueError(
                    "Vector must have shape ({},) or (N,{})".format(self.dim, self.dim))

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        return cls.exp(angle_in_radians)

    def to_angle(self):
        """Recover the rotation angle in rad from the rotation matrix."""
        return self.log()

    def cuda(self, **kwargs):
        """Return a copy with the underlying tensor on the GPU."""
        return self.__class__(self.mat.cuda(**kwargs))

    def cpu(self):
        """Return a copy with the underlying tensor on the CPU."""
        return self.__class__(self.mat.cpu())


class SE2(base.SpecialEuclideanGroup):
    pass


class SO3(base.SpecialOrthogonalGroup):
    pass


class SE3(base.SpecialEuclideanGroup):
    pass
