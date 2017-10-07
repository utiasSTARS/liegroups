class LieGroup:
    """Base class for Lie groups"""
    dim = None
    """Dimension of the transformation matrix."""
    dof = None
    """Underlying degrees of freedom (i.e., dimension of the tangent space)."""

    def __init__(self):
        raise AttributeError("Not implemented")

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a transformation from a matrix (safe, but slower)."""
        raise AttributeError("Not implemented")

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        raise AttributeError("Not implemented")

    @classmethod
    def identity(cls):
        """Return the identity transformation."""
        raise AttributeError("Not implemented")

    @classmethod
    def wedge(cls, vec):
        """wedge operator as defined by Barfoot.

        This is the inverse operation to vee()
        """
        raise AttributeError("Not implemented")

    @classmethod
    def vee(cls, mat):
        """vee operator as defined by Barfoot.

        This is the inverse operation to wedge.
        """
        raise AttributeError("Not implemented")

    @classmethod
    def left_jacobian(cls, vec):
        """Left Jacobian for the group."""
        raise AttributeError("Not implemented")

    @classmethod
    def inv_left_jacobian(cls, vec):
        """Inverse of the left Jacobian for the group."""
        raise AttributeError("Not implemented")

    @classmethod
    def exp(cls, vec):
        """Exponential map for the group.

        Computes a transformation from a tangent vector.

        This is the inverse operation to log.
        """
        raise AttributeError("Not implemented")

    def log(self):
        """Logarithmic map for the group.

        Computes a tangent vector from a transformation.

        This is the inverse operation to exp.
        """
        raise AttributeError("Not implemented")

    def inv(self):
        """Return the inverse transformation."""
        raise AttributeError("Not implemented")

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        raise AttributeError("Not implemented")

    def normalize(self):
        """Normalize the transformation matrix to ensure it is valid and
        negate the effect of rounding errors.
        """
        raise AttributeError("Not implemented")

    def dot(self, other):
        """Multiply another group element or one or more vectors on the left.
        """
        raise AttributeError("Not implemented")

    def perturb(self, vec):
        """Perturb the transformation on the left by a vector in its local tangent space.
        """
        raise AttributeError("Not implemented")

    def as_matrix(self):
        """Return the matrix representation of the transformation."""
        raise AttributeError("Not implemented")

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.as_matrix()).replace("\n", "\n| ")


class SpecialOrthogonalGroup(LieGroup):
    def __init__(self, mat):
        self.mat = mat
        """Storage for the transformation matrix"""

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

    def inv(self):
        return self.__class__(self.mat.transpose())

    def perturb(self, phi):
        self.mat = self.__class__.exp(phi).dot(self).mat

    def as_matrix(self):
        return self.mat


class SpecialEuclideanGroup(LieGroup):
    RotationType = SpecialOrthogonalGroup
    """Rotation type."""

    def __init__(self, rot, trans):
        """Create a transformation from a translation and a rotation (unsafe, but faster)."""
        self.rot = rot
        """Storage for the rotation matrix"""
        self.trans = trans
        """Storage for the translation vector"""

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
