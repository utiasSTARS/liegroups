import numpy as np


class SO3:
    """ Rotation matrix in SO(3) using active (alibi) transformations."""

    dim = 3
    """Dimension of the rotation matrix."""

    dof = 3
    """Underlying degrees of freedom (i.e., dim of the tangent space)."""

    def __init__(self, mat=np.identity(dim)):
        """Create a SO3 object from a 3x3 rotation matrix (unsafe, but faster)."""
        if mat.shape != (self.dim, self.dim):
            raise ValueError(
                "mat must have shape ({},{})".format(self.dim, self.dim))
        self.mat = mat
        """Storage for the rotation matrix"""

    @classmethod
    def from_matrix(cls, mat):
        """Create a SO3 object from a 3x3 rotation matrix (safe, but slower)."""
        if not SO3.is_valid_matrix(mat):
            raise ValueError("Invalid rotation matrix")

        return cls(mat)

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid rotation matrix."""
        return mat.shape == (cls.dim, cls.dim) and \
            np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(cls.dim))

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls(np.identity(cls.dim))

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
    def from_rpy(cls, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles."""
        return cls.rotz(yaw) * cls.roty(pitch) * cls.rotx(roll)

    def to_rpy(self):
        """Convert a rotation matrix to RPY Euler angles."""
        pitch = np.arctan2(-self.mat[2, 0],
                           np.sqrt(self.mat[0, 0]**2 + self.mat[1, 0]**2))

        if np.isclose(pitch, np.pi / 2.):
            yaw = 0.
            roll = np.arctan2(self.mat[0, 1], self.mat[1, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            yaw = 0.
            roll = -np.arctan2(self.mat[0, 1], self.mat[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(self.mat[1, 0] * sec_pitch,
                             self.mat[0, 0] * sec_pitch)
            roll = np.arctan2(self.mat[2, 1] * sec_pitch,
                              self.mat[2, 2] * sec_pitch)

        return roll, pitch, yaw

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if not np.isclose(np.linalg.norm(quat), 1.):
            raise ValueError("Quaternion must be unit length")

        if ordering is 'xyzw':
            qx, qy, qz, qw = quat
        elif ordering is 'wxyz':
            qw, qx, qy, qz = quat
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        # Form the matrix
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        R00 = 1. - 2. * (qy2 + qz2)
        R01 = 2. * (qx * qy - qw * qz)
        R02 = 2. * (qw * qy + qx * qz)

        R10 = 2. * (qw * qz + qx * qy)
        R11 = 1. - 2. * (qx2 + qz2)
        R12 = 2. * (qy * qz - qw * qx)

        R20 = 2. * (qx * qz - qw * qy)
        R21 = 2. * (qw * qx + qy * qz)
        R22 = 1. - 2. * (qx2 + qy2)

        return cls(np.array([[R00, R01, R02],
                             [R10, R11, R12],
                             [R20, R21, R22]]))

    def to_quaternion(self, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        R = self.mat
        qw = 0.5 * np.sqrt(1. + R[0, 0] + R[1, 1] + R[2, 2])

        if np.isclose(qw, 0.):
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                d = 2. * np.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / d
                qx = 0.25 * d
                qy = (R[1, 0] + R[0, 1]) / d
                qz = (R[0, 2] + R[2, 0]) / d
            elif R[1, 1] > R[2, 2]:
                d = 2. * np.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / d
                qx = (R[1, 0] + R[0, 1]) / d
                qy = 0.25 * d
                qz = (R[2, 1] + R[1, 2]) / d
            else:
                d = 2. * np.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / d
                qx = (R[0, 2] + R[2, 0]) / d
                qy = (R[2, 1] + R[1, 2]) / d
                qz = 0.25 * d
        else:
            d = 4. * qw
            qx = (R[2, 1] - R[1, 2]) / d
            qy = (R[0, 2] - R[2, 0]) / d
            qz = (R[2, 1] - R[1, 2]) / d

        # Check ordering last
        if ordering is 'xyzw':
            quat = np.array([qx, qy, qz, qw])
        elif ordering is 'wxyz':
            quat = np.array([qw, qx, qy, qz])
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        return quat

    @classmethod
    def wedge(cls, phi):
        """SO(3) wedge operator as defined by Barfoot.

        This is the inverse operation to SO3.vee.
        """
        phi = np.atleast_2d(phi)
        if phi.shape[1] != cls.dof:
            raise ValueError(
                "phi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Phi = np.zeros([phi.shape[0], cls.dim, cls.dim])
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return np.squeeze(Phi)

    @classmethod
    def vee(cls, Phi):
        """SO(3) vee operator as defined by Barfoot.

        This is the inverse operation to SO3.wedge.
        """
        if Phi.ndim < 3:
            Phi = np.expand_dims(Phi, axis=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Phi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        phi = np.empty([Phi.shape[0], cls.dim])
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return np.squeeze(phi)

    @classmethod
    def left_jacobian(cls, phi):
        """Left SO(3) Jacobian (see Barfoot).
        """
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near angle is close to 0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dim) + 0.5 * cls.wedge(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return (s / angle) * np.identity(cls.dim) + \
            (1 - s / angle) * np.outer(axis, axis) + \
            ((1 - c) / angle) * SO3.wedge(axis)

    @classmethod
    def inv_left_jacobian(cls, phi):
        """Inverse left SO(3) Jacobian (see Barfoot).
        """
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near angle is close to 0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dim) - 0.5 * cls.wedge(phi)

        axis = phi / angle
        half_angle = 0.5 * angle
        cot_half_angle = 1. / np.tan(half_angle)

        return half_angle * cot_half_angle * np.identity(cls.dim) + \
            (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - \
            half_angle * SO3.wedge(axis)

    @classmethod
    def exp(cls, phi):
        """Exponential map for SO(3).

        Computes a rotation matrix from an axis-angle tangent vector.

        This is the inverse operation to SO3.log.
        """
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near angle is close to 0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return cls(np.identity(cls.dim) + cls.wedge(phi))

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return cls(c * np.identity(cls.dim) +
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

    def perturb(self, phi):
        """Perturb the rotation on the left
        by a vector in its local tangent space."""
        perturbed = SO3.exp(phi) * self
        self.mat = perturbed.mat

    def as_matrix(self):
        """Return the 3x3 matrix representation of the rotation."""
        return self.mat

    def normalize(self):
        """ Normalize the rotation matrix to ensure it is a valid rotation and
        negate the effect of rounding errors.
        """
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        u, s, v = np.linalg.svd(self.mat, full_matrices=False)

        middle = np.identity(self.dim)
        middle[2, 2] = np.linalg.det(u) * np.linalg.det(v)

        self.mat = u.dot(middle).dot(v)

    def inv(self):
        """Return the inverse rotation."""
        return SO3(self.mat.T)

    def adjoint(self):
        """Return the adjoint matrix of the rotation."""
        return self.mat

    def dot(self, other):
        """Transform another rotation or one or more vectors.
        The multiplication operator is equivalent to dot.
        """
        if isinstance(other, SO3):
            # Compound with another rotation
            return SO3(np.dot(self.mat, other.mat))
        else:
            other = np.atleast_2d(other)

            # Transform one or more 3-vectors or fail
            if other.shape[1] == self.dim:
                return np.squeeze(np.dot(self.mat, other.T).T)
            else:
                raise ValueError("Vector must have shape ({},) or (N,{})".format(
                    self.dim, self.dim))

    def __mul__(self, other):
        return self.dot(other)

    def __repr__(self):
        return "SO3({})".format(str(self.as_matrix()).replace('\n', '\n    '))
