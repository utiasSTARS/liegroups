import numpy as np

from liegroups import SO2


class SE2:
    """Homogeneous transformation matrix in SE(2)"""

    dim = 3
    """Dimension of the rotation matrix."""

    dof = 3
    """Underlying degrees of freedom (i.e., dim of the tangent space)."""

    def __init__(self, rot=SO2.identity(), trans=np.zeros(dim - 1)):
        """Create a SE3 object from a translation and a
         rotation."""
        if not isinstance(rot, SO2):
            raise ValueError("rot must be SO2")

        if trans.size != self.dim - 1:
            raise ValueError("trans must have size 2")

        self.rot = rot
        """Storage for the rotation matrix"""

        self.trans = trans
        """Storage for the translation vector"""

    @classmethod
    def from_matrix(cls, mat):
        """Create a SE2 object from a 3x3 transformation matrix."""
        if not SE2.is_valid_matrix(mat):
            raise ValueError("Invalid transformation matrix")

        return cls(SO2(mat[0:2, 0:2]), mat[0:2, 2])

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[2, :], np.array([0, 0, 1])) and \
            SO2.is_valid_matrix(mat[0:2, 0:2])

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls.from_matrix(np.identity(cls.dim))

    @classmethod
    def wedge(cls, xi):
        """SE(2) wedge operator as defined by Barfoot.

        This is the inverse operation to SE2.vee.
        """
        if xi.size != cls.dof:
            raise ValueError("xi must have size 3")

        return np.vstack(
            [np.hstack([SO2.wedge(xi[2]),
                        np.reshape(xi[0:2], (2, 1))]),
             [0, 0, 0]]
        )

    @classmethod
    def vee(cls, Xi):
        """SE(2) vee operator as defined by Barfoot.

        This is the inverse operation to SE2.wedge.
        """
        if Xi.shape != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape (3,3)")

        return np.hstack([Xi[0:2, 2], SO2.vee(Xi[0:2, 0:2])])

    @classmethod
    def exp(cls, xi):
        """Exponential map for SE(2).

        Computes a transformation matrix from SE(2)
        tangent vector.

        This is the inverse operation to SE2.log.
        """
        if xi.size != cls.dof:
            raise ValueError("xi must have size 3")

        rho = xi[0:2]
        phi = xi[2]
        return cls(SO2.exp(phi), SO2.left_jacobian(phi).dot(rho))

    def log(self):
        """Logarithmic map for SE(2)

        Computes a SE(2) tangent vector from a transformation
        matrix.

        This is the inverse operation to SE2.exp.
        """
        phi = SO2.log(self.rot)
        rho = SO2.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def perturb(self, xi):
        """Perturb the transformation on the left
        by a vector in its local tangent space."""
        perturbed = SE2.exp(xi) * self
        self.rot = perturbed.rot
        self.trans = perturbed.trans

    def as_matrix(self):
        """Return the 3x3 matrix representation of the
        transformation."""
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (2, 1))
        return np.vstack([np.hstack([R, t]),
                          np.array([0, 0, 1])])

    def normalize(self):
        """ Normalize the rotation matrix to ensure it is a valid rotation and
        negate the effect of rounding errors.
        """
        self.rot.normalize()

    def inv(self):
        """Return the inverse transformation."""
        inv_rot = self.rot.inv()
        inv_trans = -(inv_rot * self.trans)
        return SE2(inv_rot, inv_trans)

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        rotpart = self.rot.as_matrix()
        transpart = np.array([self.trans[1], -self.trans[0]]).reshape((2, 1))
        return np.vstack([np.hstack([rotpart, transpart]),
                          [0, 0, 1]]
                         )

    def __mul__(self, other):
        if isinstance(other, SE2):
            # Compound with another transformation
            return SE2(self.rot * other.rot,
                       self.rot * other.trans + self.trans)
        elif other.size == self.dim - 1:
            # Transform a 2-vector
            return self.rot * other + self.trans
        else:
            # Transform one or more 3-vectors or fail
            return self.as_matrix().dot(other)

    def __repr__(self):
        return "SE(2) Transformation Matrix \n %s" % self.as_matrix()
