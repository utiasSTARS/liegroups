import numpy as np

from .. import base


class SpecialOrthogonalBaseNumpy(base.SpecialOrthogonalBase):
    """Implementation of methods common to SO(N) using Numpy"""

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid or normalize:
            result = cls(mat)
            if not mat_is_valid and normalize:
                result.normalize()
        else:
            raise ValueError(
                "Invalid rotation matrix. Use normalize=True to handle rounding errors.")

        return result

    @classmethod
    def is_valid_matrix(cls, mat):
        return mat.shape == (cls.dim, cls.dim) and \
            np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(cls.dim))

    @classmethod
    def identity(cls):
        return cls(np.identity(cls.dim))

    def normalize(self):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(self.mat, full_matrices=False)

        S = np.identity(self.dim)
        S[self.dim - 1, self.dim - 1] = np.linalg.det(U) * np.linalg.det(V)

        self.mat = U.dot(S).dot(V)

    def dot(self, other):
        if isinstance(other, self.__class__):
            # Compound with another rotation
            return self.__class__(np.dot(self.mat, other.mat))
        else:
            other = np.atleast_2d(other)

            # Transform one or more 2-vectors or fail
            if other.shape[1] == self.dim:
                return np.squeeze(np.dot(self.mat, other.T).T)
            else:
                raise ValueError(
                    "Vector must have shape ({},) or (N,{})".format(self.dim, self.dim))


class SpecialEuclideanBaseNumpy(base.SpecialEuclideanBase):
    """Implementation of methods common to SE(N) using Numpy"""

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid or normalize:
            result = cls(
                cls.RotationType(mat[0:cls.dim - 1, 0:cls.dim - 1]),
                mat[0:cls.dim - 1, cls.dim - 1])
            if not mat_is_valid and normalize:
                result.normalize()
        else:
            raise ValueError(
                "Invalid transformation matrix. Use normalize=True to handle rounding errors.")

        return result

    @classmethod
    def is_valid_matrix(cls, mat):
        bottom_row = np.append(np.zeros(cls.dim - 1), 1.)

        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[cls.dim - 1, :], bottom_row) and \
            cls.RotationType.is_valid_matrix(mat[0:cls.dim - 1, 0:cls.dim - 1])

    @classmethod
    def identity(cls):
        return cls.from_matrix(np.identity(cls.dim))

    def as_matrix(self):
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (self.dim - 1, 1))
        bottom_row = np.append(np.zeros(self.dim - 1), 1.)
        return np.vstack([np.hstack([R, t]),
                          bottom_row])

    def dot(self, other):
        if isinstance(other, self.__class__):
            # Compound with another transformation
            return self.__class__(self.rot.dot(other.rot),
                                  self.rot.dot(other.trans) + self.trans)
        else:
            other = np.atleast_2d(other)

            if other.shape[1] == self.dim - 1:
                # Transform one or more 2-vectors
                return np.squeeze(self.rot.dot(other) + self.trans)
            elif other.shape[1] == self.dim:
                # Transform one or more 3-vectors
                return np.squeeze(self.as_matrix().dot(other.T)).T
            else:
                raise ValueError("Vector must have shape ({},), ({},), (N,{}) or (N,{})".format(
                    self.dim - 1, self.dim, self.dim - 1, self.dim))
