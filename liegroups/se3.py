import numpy as np

from liegroups.so3 import SO3


class SE3:
    """Homogeneous transformation matrix in SE(3)

    T = [[T.rot T.trans]
         [    0       1]]
    """

    def __init__(self, rot=SO3.identity(), trans=np.zeros(3)):
        """Create a SE3 object from a translation and a
         rotation."""
        if not isinstance(rot, SO3):
            raise ValueError("rot must be SO3")

        if trans.size != 3:
            raise ValueError("trans must have size 3")

        self.rot = rot
        self.trans = trans

    @classmethod
    def from_matrix(cls, mat):
        """Create a SE3 object from a 4x4 transformation matrix."""
        if not SE3.is_valid_matrix(mat):
            raise ValueError("Invalid transformation matrix")

        return cls(SO3(mat[0:3, 0:3]), mat[0:3, 3])

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        return mat.shape == (4, 4) and \
            np.array_equal(mat[3, :], np.array([0, 0, 0, 1])) and \
            SO3.is_valid_matrix(mat[0:3, 0:3])

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls.from_matrix(np.identity(4))

    @classmethod
    def wedge(cls, xi):
        """SE(3) wedge operator as defined by Barfoot.

        This is the inverse operation to SE3.vee.
        """
        if xi.size != 6:
            raise ValueError("xi must have size 6")

        return np.vstack(
            [np.hstack([SO3.wedge(xi[3:7]),
                        np.reshape(xi[0:3], (3, 1))]),
             [0, 0, 0, 0]]
        )

    @classmethod
    def vee(cls, Xi):
        """SE(3) vee operator as defined by Barfoot.

        This is the inverse operation to SE3.wedge.
        """
        if Xi.shape != (4, 4):
            raise ValueError("Xi must have shape (4,4)")

        return np.hstack([Xi[0:3, 3], SO3.vee(Xi[0:3, 0:3])])

    @classmethod
    def exp(cls, xi):
        """Exponential map for SE(3).

        Computes a transformation matrix from SE(3)
        tangent vector.

        This is the inverse operation to SE3.log.
        """
        if xi.size != 6:
            raise ValueError("xi must have size 6")

        rho = xi[0:3]
        phi = xi[3:6]
        return cls(SO3.exp(phi), SO3.left_jacobian(phi).dot(rho))

    def log(self):
        """Logarithmic map for SE(3)

        Computes a SE(3) tangent vector from a transformation
        matrix.

        This is the inverse operation to SE3.exp.
        """
        phi = SO3.log(self.rot)
        rho = SO3.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def as_matrix(self):
        """Return the 4x4 matrix representation of the
        transformation."""
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (3, 1))
        return np.vstack([np.hstack([R, t]),
                          np.array([0, 0, 0, 1])])

    def normalize(self):
        """ Normalize the rotation matrix to ensure it is a valid rotation and
        negate the effect of rounding errors.
        """
        self.rot.normalize()

    def inv(self):
        """Return the inverse transformation."""
        inv_rot = self.rot.inv()
        inv_trans = -(inv_rot * self.trans)
        return SE3(inv_rot, inv_trans)

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        rotmat = self.rot.as_matrix()
        return np.vstack(
            [np.hstack([rotmat,
                        SO3.wedge(self.trans).dot(rotmat)]),
             np.hstack([np.zeros((3, 3)), rotmat])]
        )

    def __mul__(self, other):
        if isinstance(other, SE3):
            # Compound with another transformation
            return SE3(self.rot * other.rot,
                       self.rot * other.trans + self.trans)
        elif other.size == 3:
            # Transform a 3-vector
            return self.rot * other + self.trans
        else:
            # Transform one or more 4-vectors or fail
            return self.as_matrix().dot(other)

    def __repr__(self):
        return "SE(3) Transformation Matrix \n %s" % self.as_matrix()
