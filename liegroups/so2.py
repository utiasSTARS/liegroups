import numpy as np


class SO2:
    """ Rotation matrix in SO(2)
    """

    def __init__(self, mat=np.identity(2)):
        """Create a SO2 object from a 2x2 rotation matrix."""
        if not SO2.isvalidmatrix(mat):
            raise ValueError("Invalid rotation matrix")

        self.mat = mat

    @classmethod
    def frommatrix(cls, mat):
        """Create a SO2 object from a 2x2 rotation matrix."""
        if not SO2.isvalidmatrix(mat):
            raise ValueError("Invalid rotation matrix")

        return cls(mat)

    @classmethod
    def isvalidmatrix(cls, mat):
        """Check if a matrix is a valid rotation matrix."""
        return mat.shape == (2, 2) and np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(2))

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls(np.identity(2))

    @classmethod
    def fromangle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[c, -s],
                             [s,  c]]))

    @classmethod
    def wedge(cls, phi):
        """SO(2) wedge operator as defined by Barfoot.

        This is the inverse operation to SO2.vee.
        """
        return np.array([[0, -phi],
                         [phi, 0]])

    @classmethod
    def vee(cls, Phi):
        """SO(2) vee operator as defined by Barfoot.

        This is the inverse operation to SO2.wedge.
        """
        if Phi.shape != (2, 2):
            raise ValueError("Phi must have shape (2,2)")

        return Phi[1, 0]

    @classmethod
    def exp(cls, phi):
        """Exponential map for SO(2).

        Computes a rotation matrix from an angle.

        This is the inverse operation to SO2.log.
        """
        return cls.fromangle(phi)

    def log(self):
        """Logarithmic map for SO(3).

        Computes an angle from a rotation matrix.

        This is the inverse operation to SO2.exp.
        """
        return np.arctan2(self.mat[1, 0], self.mat[0, 0])

    def asmatrix(self):
        """Return the 2x2 matrix representation of the rotation."""
        return self.mat

    def normalize(self):
        """ Normalize the rotation matrix to ensure it is a valid rotation and
        negate the effect of rounding errors.
        """
        U, s, V = np.linalg.svd(self.mat, full_matrices=False)

        middle = np.identity(2)
        middle[1, 1] = np.linalg.det(V) * np.linalg.det(U)

        self.mat = U.dot(middle.dot(V.T))

    def inverse(self):
        """Return the inverse rotation."""
        return SO2(self.mat.T)

    def adjoint(self):
        """Return the adjoint matrix of the rotation."""
        return 1.

    def __mul__(self, other):
        if isinstance(other, SO2):
            # Compound with another rotation
            return SO2(np.dot(self.mat, other.mat))
        else:
            # Transform one or more 2-vectors or fail
            return np.dot(self.mat, other)

    def __repr__(self):
        return "SO(2) Rotation Matrix \n %s" % self.asmatrix()
