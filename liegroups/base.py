from abc import ABC, abstractmethod


class LieGroupBase(ABC):
    """Base class for Lie groups."""
    @property
    @classmethod
    @abstractmethod
    def dim(cls):
        """Dimension of the transformation matrix."""
        pass

    @property
    @classmethod
    @abstractmethod
    def dof(cls):
        """Underlying degrees of freedom (i.e., dimension of the tangent space)."""
        pass

    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a transformation from a matrix (safe, but slower)."""
        pass

    @classmethod
    @abstractmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        pass

    @classmethod
    @abstractmethod
    def identity(cls):
        """Return the identity transformation."""
        pass

    @classmethod
    @abstractmethod
    def wedge(cls, vec):
        """wedge operator as defined by Barfoot.

        This is the inverse operation to vee()
        """
        pass

    @classmethod
    @abstractmethod
    def vee(cls, mat):
        """vee operator as defined by Barfoot.

        This is the inverse operation to wedge.
        """
        pass

    @classmethod
    @abstractmethod
    def left_jacobian(cls, vec):
        """Left Jacobian for the group."""
        pass

    @classmethod
    @abstractmethod
    def inv_left_jacobian(cls, vec):
        """Inverse of the left Jacobian for the group."""
        pass

    @classmethod
    @abstractmethod
    def exp(cls, vec):
        """Exponential map for the group.

        Computes a transformation from a tangent vector.

        This is the inverse operation to log.
        """
        pass

    @abstractmethod
    def log(self):
        """Logarithmic map for the group.

        Computes a tangent vector from a transformation.

        This is the inverse operation to exp.
        """
        pass

    @abstractmethod
    def inv(self):
        """Return the inverse transformation."""
        pass

    @abstractmethod
    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        pass

    @abstractmethod
    def normalize(self):
        """Normalize the transformation matrix to ensure it is valid and
        negate the effect of rounding errors.
        """
        pass

    @abstractmethod
    def dot(self, other):
        """Multiply another group element or one or more vectors on the left.
        """
        pass

    @abstractmethod
    def perturb(self, vec):
        """Perturb the transformation on the left by a vector in its local tangent space.
        """
        pass

    @abstractmethod
    def as_matrix(self):
        """Return the matrix representation of the transformation."""
        pass

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.as_matrix()).replace("\n", "\n| ")


class SpecialOrthogonalBase(LieGroupBase, ABC):
    """Base class for Special Orthogonal groups SO(N)."""

    def __init__(self, mat):
        self.mat = mat
        """Storage for the transformation matrix."""

    def inv(self):
        if len(self.mat.shape) < 3:
            return self.__class__(self.mat.transpose(1, 0))
        else:
            return self.__class__(self.mat.transpose(2, 1))

    def perturb(self, phi):
        self.mat = self.__class__.exp(phi).dot(self).mat

    def as_matrix(self):
        return self.mat


class SpecialEuclideanBase(LieGroupBase, ABC):
    """Base class for Special Euclidean groups SE(N)."""
    @property
    @classmethod
    @abstractmethod
    def RotationType(cls):
        """Rotation type."""
        pass

    def __init__(self, rot, trans):
        """Create a transformation from a translation and a rotation (unsafe, but faster)"""
        self.rot = rot
        """Storage for the rotation matrix."""
        self.trans = trans
        """Storage for the translation vector."""

    def normalize(self):
        self.rot.normalize()

    def inv(self):
        inv_rot = self.rot.inv()
        inv_trans = -(inv_rot.dot(self.trans))
        return self.__class__(inv_rot, inv_trans)

    def perturb(self, xi):
        perturbed = self.__class__.exp(xi).dot(self)
        self.rot = perturbed.rot
        self.trans = perturbed.trans
