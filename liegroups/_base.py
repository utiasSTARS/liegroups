from abc import ABCMeta, abstractmethod

# support for both python 2 and 3
from future.utils import with_metaclass


class LieGroupBase(with_metaclass(ABCMeta)):
    """ Common abstract base class defining basic interface for Lie groups.
        Does not depend on any specific linear algebra library.
    """

    def __init__(self):
        pass

    @property
    @classmethod
    @abstractmethod
    def dim(cls):
        """Dimension of the underlying representation."""
        pass

    @property
    @classmethod
    @abstractmethod
    def dof(cls):
        """Underlying degrees of freedom (i.e., dimension of the tangent space)."""
        pass

    @abstractmethod
    def dot(self, other):
        """Multiply another group element or one or more vectors on the left.
        """
        pass

    @classmethod
    @abstractmethod
    def exp(cls, vec):
        """Exponential map for the group.

        Computes a transformation from a tangent vector.

        This is the inverse operation to log.
        """
        pass

    @classmethod
    @abstractmethod
    def identity(cls):
        """Return the identity transformation."""
        pass

    @abstractmethod
    def inv(self):
        """Return the inverse transformation."""
        pass

    @abstractmethod
    def log(self):
        """Logarithmic map for the group.

        Computes a tangent vector from a transformation.

        This is the inverse operation to exp.
        """
        pass

    @abstractmethod
    def normalize(self):
        """Normalize the group element to ensure it is valid and
        negate the effect of rounding errors.
        """
        pass

    @abstractmethod
    def perturb(self, vec):
        """Perturb the group element on the left by a vector in its local tangent space.
        """
        pass


class MatrixLieGroupBase(LieGroupBase):
    """Common abstract base class defining basic interface for Matrix Lie Groups.
       Does not depend on any specific linear algebra library.
    """

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.as_matrix()).replace("\n", "\n| ")

    @abstractmethod
    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        pass

    @abstractmethod
    def as_matrix(self):
        """Return the matrix representation of the transformation."""
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a transformation from a matrix (safe, but slower)."""
        pass

    @classmethod
    @abstractmethod
    def inv_left_jacobian(cls, vec):
        """Inverse of the left Jacobian for the group."""
        pass

    @classmethod
    @abstractmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        pass

    @classmethod
    @abstractmethod
    def left_jacobian(cls, vec):
        """Left Jacobian for the group."""
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
    def wedge(cls, vec):
        """wedge operator as defined by Barfoot.

        This is the inverse operation to vee.
        """
        pass


class SOMatrixBase(MatrixLieGroupBase):
    """Common abstract base class for Special Orthogonal Matrix Lie Groups SO(N).
       Does not depend on any specific linear algebra library.
    """

    def __init__(self, mat):
        """Create a transformation from a rotation matrix (unsafe, but faster)."""
        self.mat = mat
        """Storage for the rotation matrix."""

    def as_matrix(self):
        """Return the matrix representation of the rotation."""
        return self.mat

    def perturb(self, phi):
        """Perturb the rotation in-place on the left by a vector in its local tangent space.

        .. math::
            \\mathbf{C} \\gets \\exp(\\boldsymbol{\\phi}^\\wedge) \\mathbf{C}
        """
        self.mat = self.__class__.exp(phi).dot(self).mat


class SEMatrixBase(MatrixLieGroupBase):
    """Common abstract base class for Special Euclidean Matrix Lie Groups SE(N).
       Does not depend on any specific linear algebra library.
    """

    def __init__(self, rot, trans):
        """Create a transformation from a translation and a rotation (unsafe, but faster)"""
        self.rot = rot
        """Storage for the rotation matrix."""
        self.trans = trans
        """Storage for the translation vector."""

    @classmethod
    @abstractmethod
    def odot(cls, p, directional=False):
        """odot operator as defined by Barfoot."""
        pass

    def perturb(self, xi):
        """Perturb the transformation in-place on the left by a vector in its local tangent space.

        .. math::
            \\mathbf{T} \\gets \\exp(\\boldsymbol{\\xi}^\\wedge) \\mathbf{T}
        """
        perturbed = self.__class__.exp(xi).dot(self)
        self.rot = perturbed.rot
        self.trans = perturbed.trans

    @property
    @classmethod
    @abstractmethod
    def RotationType(cls):
        """Rotation type."""
        pass


class VectorLieGroupBase(LieGroupBase):
    """Common abstract base class for Lie Groups with vector parametrizations 
       (complex, quaternions, dual quaternions). Does not depend on any  
       specific linear algebra library.
    """

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.data).replace("\n", "\n| ")

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the vector"""
        pass
