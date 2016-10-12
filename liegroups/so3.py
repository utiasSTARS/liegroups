import numpy as np


class SO3:
    """ Rotation matrix in SO(3)
    """

    def __init__(self, mat=np.identity(3)):
        """Create a SO3 object from a 3x3 rotation matrix."""
        if not SO3.isvalidmatrix(mat):
            raise ValueError("Invalid rotation matrix")

        self.mat = mat

    @classmethod
    def frommatrix(cls, mat):
        """Create a SO3 object from a 3x3 rotation matrix."""
        if not SO3.isvalidmatrix(mat):
            raise ValueError("Invalid rotation matrix")

        return cls(mat)

    @classmethod
    def isvalidmatrix(cls, mat):
        """Check if a matrix is a valid rotation matrix."""
        return mat.shape == (3, 3) and np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(3))

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls(np.identity(3))

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis."""
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[1., 0., 0.],
                             [0., c, -s],
                             [0., s,  c]]))

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis."""
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[c,  0., s],
                             [0., 1., 0.],
                             [-s, 0., c]]))

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis."""
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[c, -s,  0.],
                             [s,  c,  0.],
                             [0., 0., 1.]]))

    @classmethod
    def from_rpy(cls, r, p, y):
        """Form a rotation matrix from RPY Euler angles."""
        return cls.rotz(y) * cls.roty(p) * cls.rotx(r)

    @classmethod
    def wedge(cls, phi):
        """SO(3) wedge operator as defined by Barfoot.

        This is the inverse operation to SO3.vee.
        """
        if phi.size != 3:
            raise ValueError("phi must have size 3")

        return np.array([[0, -phi[2], phi[1]],
                         [phi[2], 0, -phi[0]],
                         [-phi[1], phi[0], 0]])

    @classmethod
    def vee(cls, Phi):
        """SO(3) vee operator as defined by Barfoot.

        This is the inverse operation to SO3.wedge.
        """
        if Phi.shape != (3, 3):
            raise ValueError("Phi must have shape (3,3)")

        return np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])

    @classmethod
    def exp(cls, phi):
        """Exponential map for SO(3).

        Computes a rotation matrix from an axis-angle tangent vector.

        This is the inverse operation to SO3.log.
        """
        if phi.size != 3:
            raise ValueError("phi must have size 3")

        angle = np.linalg.norm(phi)

        # Near angle is close to 0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(3) + cls.wedge(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return cls(c * np.identity(3) +
                   (1 - c) * np.outer(axis, axis) +
                   s * cls.wedge(axis))

    def log(self):
        """Logarithmic map for SO(3).

        Computes an axis-angle tangent vector from a rotation matrix.

        This is the inverse operation to SO3.exp.
        """

        # The rotation axis (not unit-length) is given by
        axis = np.array([self.mat[2, 1] - self.mat[1, 2],
                         self.mat[0, 2] - self.mat[2, 0],
                         self.mat[1, 0] - self.mat[0, 1]])

        # The sine of the rotation angle is half the norm of the axis
        sin_angle = 0.5 * np.linalg.norm(axis)

        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(self.mat) - 0.5

        angle = np.arctan2(sin_angle, cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.):
            return SO3.vee(self.mat - np.identity(3))

        # Otherwise normalize the axis and return the axis-angle vector
        return 0.5 * angle * axis / sin_angle

    def asmatrix(self):
        """Return the 3x3 matrix representation of the rotation."""
        return self.mat

    def normalize(self):
        """ Normalize the rotation matrix to ensure it is a valid rotation and
        negate the effect of rounding errors.
        """
        U, s, V = np.linalg.svd(self.mat, full_matrices=False)

        middle = np.identity(3)
        middle[2, 2] = np.linalg.det(V) * np.linalg.det(U)

        self.mat = U.dot(middle.dot(V.T))

    def inverse(self):
        """Return the inverse rotation."""
        return SO3(self.mat.T)

    def adjoint(self):
        """Return the adjoint matrix of the rotation."""
        return self.mat

    def __mul__(self, other):
        if isinstance(other, SO3):
            # Compound with another rotation
            return SO3(np.dot(self.mat, other.mat))
        else:
            # Transform one or more 3-vectors or fail
            return np.dot(self.mat, other)

    def __repr__(self):
        return "SO(3) Rotation Matrix \n %s" % self.asmatrix()
