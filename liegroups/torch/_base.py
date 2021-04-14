import torch
import numpy as np  # for matrix determinant and SVD

from .. import _base
from . import utils


class SOMatrixBase(_base.SOMatrixBase):
    """Implementation of methods common to SO(N) matrix lie groups using PyTorch"""

    def cpu(self):
        """Return a copy with the underlying tensor on the CPU."""
        return self.__class__(self.mat.cpu())

    def cuda(self, device=None, non_blocking=False):
        """Return a copy with the underlying tensor on the GPU."""
        return self.__class__(self.mat.cuda(device=device, non_blocking=non_blocking))

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

            # Transform one or more vectors or fail
            if other.shape[0] != mat.shape[0]:
                raise ValueError("Expected vector-batch batch size of {}, got {}".format(
                    mat.shape[0], other.shape[0]))

            if other.shape[2] == self.dim:
                return torch.bmm(mat, other.transpose(2, 1)).transpose_(2, 1).squeeze_()
            else:
                raise ValueError(
                    "Vector or vector-batch must have shape ({},), (N,{}), or ({},N,{})".format(self.dim, self.dim, mat.shape[0], self.dim))

    @classmethod
    def from_matrix(cls, mat, normalize=False):

        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid.all() or normalize:
            result = cls(mat)

            if normalize:
                result.normalize(inds=mat_is_valid.logical_not().nonzero(as_tuple=False))

            return result
        else:
            raise ValueError(
                "Invalid rotation matrix. Use normalize=True to handle rounding errors.")

    @classmethod
    def from_numpy(cls, other, pin_memory=False):
        """Create a torch-based copy of a numpy-based rotation."""
        mat = torch.Tensor(other.mat)
        if pin_memory:
            mat = mat.pin_memory()

        return cls(mat)

    @classmethod
    def identity(cls, batch_size=1, copy=False):
        if copy:
            mat = torch.eye(cls.dim).repeat(batch_size, 1, 1)
        else:
            mat = torch.eye(cls.dim).expand(
                batch_size, cls.dim, cls.dim).squeeze()
        return cls(mat)

    def inv(self):
        if self.mat.dim() < 3:
            return self.__class__(self.mat.transpose(1, 0))
        else:
            return self.__class__(self.mat.transpose(2, 1))

    def is_cuda(self):
        """Returns true if the underlying tensor is a CUDA tensor"""
        return self.mat.is_cuda

    def is_pinned(self):
        """Returns true if the underlying tensor resides in pinned memory"""
        return self.mat.is_pinned()

    @classmethod
    def is_valid_matrix(cls, mat):
        if mat.dim() < 3:
            mat = mat.unsqueeze(dim=0)

        # Check the shape
        if mat.is_cuda:
            shape_check = torch.cuda.BoolTensor(mat.shape[0]).fill_(False)
        else:
            shape_check = torch.BoolTensor(mat.shape[0]).fill_(False)

        if mat.shape[1:3] != (cls.dim, cls.dim):
            return shape_check
        else:
            shape_check.fill_(True)

        # Determinants of each matrix in the batch should be 1
        det_check = utils.isclose(mat.__class__(
            np.linalg.det(mat.detach().cpu().numpy())), 1.)

        # The transpose of each matrix in the batch should be its inverse
        inv_check = utils.isclose(mat.transpose(2, 1).bmm(mat),
                                  torch.eye(cls.dim, dtype=mat.dtype)).sum(dim=1).sum(dim=1) \
            == cls.dim * cls.dim

        return shape_check & det_check & inv_check

    def _normalize_one(self, mat):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).

        # pytorch has native SVD function but not determinant...
        # U, _, V = mat.squeeze().svd()
        # S = torch.eye(self.dim)
        # if U.is_cuda:
        #     S = S.cuda()
        # S[self.dim - 1, self.dim - 1] = float(np.linalg.det(U.cpu().numpy()) *
        #                                       np.linalg.det(V.cpu().numpy()))
        # mat_normalized = U.mm(S).mm(V.t_())

        # pytorch SVD seems to be inaccurate, so just move to numpy immediately
        mat_cpu = mat.detach().cpu().numpy().squeeze()
        U, _, V = np.linalg.svd(mat_cpu, full_matrices=False)
        S = np.eye(self.dim)
        S[self.dim - 1, self.dim - 1] = np.linalg.det(U) * np.linalg.det(V)

        mat_normalized = mat.__class__(U.dot(S).dot(V))

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

    def pin_memory(self):
        """Return a copy with the underlying tensor in pinned (page-locked) memory. Makes host-to-GPU copies faster.

        See: http://pytorch.org/docs/master/notes/cuda.html?highlight=pinned
        """
        return self.__class__(self.mat.pin_memory())


class SEMatrixBase(_base.SEMatrixBase):
    """Implementation of methods common to SE(N) matrix lie groups using PyTorch"""

    def __init__(self, rot, trans):
        super(SEMatrixBase, self).__init__(rot, trans)

    def as_matrix(self):
        R = self.rot.as_matrix()
        if R.dim() < 3:
            R = R.unsqueeze(dim=0)

        if self.trans.dim() < 2:
            t = self.trans.unsqueeze(dim=0)
        else:
            t = self.trans

        t = t.unsqueeze(dim=2)  # N x self.dim-1 x 1

        bottom_row = self.trans.new_zeros(self.dim)
        bottom_row[-1] = 1.
        bottom_row = bottom_row.unsqueeze_(dim=0).unsqueeze_(
            dim=1).expand(R.shape[0], 1, self.dim)

        return torch.cat([torch.cat([R, t], dim=2),
                          bottom_row], dim=1).squeeze_()

    def cpu(self):
        """Return a copy with the underlying tensors on the CPU."""
        return self.__class__(self.rot.cpu(), self.trans.cpu())

    def cuda(self, device=None, non_blocking=False):
        """Return a copy with the underlying tensors on the GPU."""
        return self.__class__(self.rot.cuda(device=device, non_blocking=non_blocking),
                              self.trans.cuda(device=device, non_blocking=non_blocking))

    def dot(self, other):
        if isinstance(other, self.__class__):
            if other.trans.dim() == 2:
                # vectorbatch --> matrixbatch (NxD --> Nx1xD)
                other_trans = other.trans.unsqueeze(dim=1)
            else:
                # vector --> matrix (D --> 1xD)
                other_trans = other.trans.unsqueeze(dim=0)

            # Compound with another transformation
            return self.__class__(self.rot.dot(other.rot),
                                  self.rot.dot(other_trans) + self.trans)
        else:
            if other.dim() < 2:
                other = other.unsqueeze(dim=0)  # vector --> matrix
            if other.dim() < 3:
                other = other.unsqueeze(dim=0)  # matrix --> batch

            # Got euclidean coordinates
            if other.shape[2] == self.dim - 1:
                rot = self.rot.as_matrix()
                trans = self.trans

                if trans.dim() < 2:
                    trans = trans.unsqueeze(dim=0)  # vector --> vectorbatch
                if trans.dim() < 3:
                    # vectorbatch --> matrixbatch
                    trans = trans.unsqueeze(dim=1)

                if rot.dim() < 3:
                    # matrix --> batch
                    rot = rot.unsqueeze(dim=0).expand(
                        other.shape[0], rot.shape[0], rot.shape[1])
                    # matrix --> batch
                    trans = trans.expand(
                        other.shape[0], trans.shape[1], trans.shape[2])
                elif other.shape[0] == 1:
                    other = other.expand(
                        rot.shape[0], other.shape[1], other.shape[2])

                # Transform one or more vectors or fail
                if other.shape[0] != rot.shape[0]:
                    raise ValueError(
                        "Expected vector-batch batch size of {}, got {}".format(rot.shape[0], other.shape[0]))

                # rot * other + trans
                return torch.baddbmm(trans.transpose(2, 1), rot,
                                     other.transpose(2, 1)
                                     ).transpose(2, 1).squeeze_()

            # Got homogeneous coordinates
            elif other.shape[2] == self.dim:
                mat = self.as_matrix()

                if mat.dim() < 3:
                    mat = mat.unsqueeze(dim=0).expand(
                        other.shape[0], self.dim, self.dim)  # matrix --> batch
                elif other.shape[0] == 1:
                    other = other.expand(
                        mat.shape[0], other.shape[1], other.shape[2])

                if other.shape[0] != mat.shape[0]:
                    raise ValueError(
                        "Expected vector-batch batch size of {}, got {}".format(mat.shape[0], other.shape[0]))

                return torch.bmm(mat, other.transpose(2, 1)
                                 ).transpose(2, 1).squeeze_()

            # Got wrong dimension
            else:
                if self.trans.dim() < 2:
                    batch_size = 1
                else:
                    batch_size = self.trans.shape[0]

                raise ValueError(
                    "Vector or vector-batch must have shape ({},), ({},), (N,{}), (N,{}), ({},N,{}), or ({},N,{})".format(self.dim - 1, self.dim, self.dim - 1, self.dim, batch_size, self.dim - 1, batch_size, self.dim))

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        if mat.dim() < 3:
            mat = mat.unsqueeze(dim=0)

        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid.all() or normalize:
            rot = mat[:, 0:cls.dim - 1, 0:cls.dim - 1].squeeze()
            trans = mat[:, 0:cls.dim - 1, cls.dim - 1].squeeze()
            result = cls(cls.RotationType(rot), trans)

            if normalize:
                result.normalize(inds=mat_is_valid.logical_not().nonzero(as_tuple=False))

            return result
        else:
            raise ValueError(
                "Invalid transformation matrix. Use normalize=True to handle rounding errors.")

    @classmethod
    def from_numpy(cls, other, pin_memory=False):
        """Create a torch-based copy of a numpy-based transformation."""
        rot = cls.RotationType.from_numpy(other.rot, pin_memory)

        trans = torch.Tensor(other.trans)
        if pin_memory:
            trans = torch.Tensor(other.trans).pin_memory

        return cls(rot, trans)

    @classmethod
    def identity(cls, batch_size=1, copy=False):
        if copy:
            mat = torch.eye(cls.dim).repeat(batch_size, 1, 1)
        else:
            mat = torch.eye(cls.dim).expand(batch_size, cls.dim, cls.dim)

        return cls.from_matrix(mat.squeeze_())

    def inv(self):
        if self.trans.dim() == 2:
            # vectorbatch --> matrixbatch (NxD --> Nx1xD)
            trans = self.trans.unsqueeze(dim=1)
        else:
            # vector --> matrix (D --> 1xD)
            trans = self.trans.unsqueeze(dim=0)

        inv_rot = self.rot.inv()
        inv_trans = -(inv_rot.dot(trans))
        return self.__class__(inv_rot, inv_trans)

    def is_cuda(self):
        """Returns true if the underlying tensors are CUDA tensors"""
        return self.rot.is_cuda()

    def is_pinned(self):
        """Returns true if the underlying tensors reside in pinned memory"""
        return self.rot.is_pinned()

    @classmethod
    def is_valid_matrix(cls, mat):
        if mat.dim() < 3:
            mat = mat.unsqueeze(dim=0)

        # Check the shape
        if mat.is_cuda:
            shape_check = torch.cuda.BoolTensor(mat.shape[0]).fill_(False)
        else:
            shape_check = torch.BoolTensor(mat.shape[0]).fill_(False)

        if mat.shape[1:3] != (cls.dim, cls.dim):
            return shape_check
        else:
            shape_check.fill_(True)

        # Bottom row should be [zeros, 1]
        bottom_row = mat.new_zeros(cls.dim)
        bottom_row[-1] = 1.
        bottom_check = (mat[:, cls.dim - 1, :] == bottom_row.unsqueeze_(
            dim=0).expand(mat.shape[0], cls.dim)).sum(dim=1) == cls.dim

        # Check that the rotation part is valid
        rot_check = cls.RotationType.is_valid_matrix(
            mat[:, 0:cls.dim - 1, 0:cls.dim - 1])

        return shape_check & bottom_check & rot_check

    def normalize(self, inds=None):
        self.rot.normalize(inds)

    def pin_memory(self):
        """Return a copy with the underlying tensor in pinned (page-locked) memory. Makes host-to-GPU copies faster.

        See: http://pytorch.org/docs/master/notes/cuda.html?highlight=pinned
        """
        return self.__class__(self.rot.pin_memory(), self.trans.pin_memory())


class VectorLieGroupBase(_base.VectorLieGroupBase):
    """Implementation of methods common to vector-parametrized lie groups using PyTorch"""
    pass
