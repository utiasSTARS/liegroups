import numpy as np

from . import base


class SO2(base.SpecialOrthogonalGroup):
    """Rotation matrix in SO(2) using active (alibi) transformations."""
    dim = 2
    dof = 1

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def is_valid_matrix(cls, mat):
        return mat.shape == (cls.dim, cls.dim) and \
            np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(cls.dim))

    @classmethod
    def identity(cls):
        return cls(np.identity(cls.dim))

    @classmethod
    def wedge(cls, phi):
        phi = np.atleast_1d(phi)

        Phi = np.zeros([len(phi), cls.dim, cls.dim])
        Phi[:, 0, 1] = -phi
        Phi[:, 1, 0] = phi
        return np.squeeze(Phi)

    @classmethod
    def vee(cls, Phi):
        if Phi.ndim < 3:
            Phi = np.expand_dims(Phi, axis=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(cls.dim, cls.dim, cls.dim, cls.dim))

        return np.squeeze(Phi[:, 1, 0])

    @classmethod
    def left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) + 0.5 * cls.wedge(phi)

        s = np.sin(phi)
        c = np.cos(phi)

        return (1. / phi) * np.array([[s, -(1 - c)],
                                      [1 - c, s]])

    @classmethod
    def inv_left_jacobian(cls, phi):
        """(see Barfoot/Eade)."""
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) - 0.5 * cls.wedge(phi)

        A = np.sin(phi) / phi
        B = (1. - np.cos(phi)) / phi
        return (1. / (A * A + B * B)) * np.array([[A, B], [-B, A]])

    @classmethod
    def exp(cls, phi):
        c = np.cos(phi)
        s = np.sin(phi)

        return cls(np.array([[c, -s],
                             [s,  c]]))

    def log(self):
        c = self.mat[0, 0]
        s = self.mat[1, 0]
        return np.arctan2(s, c)

    def adjoint(self):
        return 1.

    def normalize(self):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(self.mat, full_matrices=False)

        S = np.identity(self.dim)
        S[1, 1] = np.linalg.det(U) * np.linalg.det(V)

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

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad."""
        return cls.exp(angle_in_radians)

    def to_angle(self):
        """Recover the rotation angle in rad from the rotation matrix."""
        return self.log()


class SE2(base.SpecialEuclideanGroup):
    """Homogeneous transformation matrix in SE(2) using active (alibi) transformations."""
    dim = 3
    dof = 3
    RotationType = SO2

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[2, :], np.array([0, 0, 1])) and \
            cls.RotationType.is_valid_matrix(mat[0:2, 0:2])

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls.from_matrix(np.identity(cls.dim))

    @classmethod
    def wedge(cls, xi):
        """SE(2) wedge operator as defined by Barfoot.

        This is the inverse operation to SE2.vee.
        """
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dof, cls.dof])
        Xi[:, 0:2, 0:2] = SO2.wedge(xi[:, 2])
        Xi[:, 0:2, 2] = xi[:, 0:2]

        return np.squeeze(Xi)

    @classmethod
    def vee(cls, Xi):
        """SE(2) vee operator as defined by Barfoot.

        This is the inverse operation to SE2.wedge.
        """
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = np.empty([Xi.shape[0], cls.dim])
        xi[:, 0:2] = Xi[:, 0:2, 2]
        xi[:, 2] = SO2.vee(Xi[:, 0:2, 0:2])
        return np.squeeze(xi)

    @classmethod
    def exp(cls, xi):
        if len(xi) != cls.dof:
            raise ValueError("xi must have length 3")

        rho = xi[0:2]
        phi = xi[2]
        return cls(SO2.exp(phi), SO2.left_jacobian(phi).dot(rho))

    def log(self):
        phi = SO2.log(self.rot)
        rho = SO2.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def as_matrix(self):
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (2, 1))
        return np.vstack([np.hstack([R, t]),
                          np.array([0, 0, 1])])

    def adjoint(self):
        rotpart = self.rot.as_matrix()
        transpart = np.array([self.trans[1], -self.trans[0]]).reshape((2, 1))
        return np.vstack([np.hstack([rotpart, transpart]),
                          [0, 0, 1]])

    def dot(self, other):
        if isinstance(other, SE2):
            # Compound with another transformation
            return SE2(self.rot.dot(other.rot),
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

    @classmethod
    def odot(cls, p, **kwargs):
        """SE(2) \odot operator as defined by Barfoot."""
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # vector, in which case the scale is 0
            scale_is_zero = kwargs.get('directional', False)
            if not scale_is_zero:
                result[:, 0:2, 0:2] = np.eye(2)

            result[:, 0:2, 2] = SO2.wedge(1).dot(p.T).T

        elif p.shape[1] == cls.dim:
            result[:, 0:2, 0:2] = p[:, 2] * np.eye(2)
            result[:, 0:2, 2] = SO2.wedge(1).dot(p[:, 0:2].T).T

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)


class SO3(base.SpecialOrthogonalGroup):
    """Rotation matrix in SO(3) using active (alibi) transformations."""
    dim = 3
    dof = 3

    def __init__(self, mat):
        super().__init__(mat)

    @classmethod
    def is_valid_matrix(cls, mat):
        return mat.shape == (cls.dim, cls.dim) and \
            np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(cls.dim))

    @classmethod
    def identity(cls):
        return cls(np.identity(cls.dim))

    @classmethod
    def wedge(cls, phi):
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
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dim) + 0.5 * cls.wedge(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return (s / angle) * np.identity(cls.dim) + \
            (1 - s / angle) * np.outer(axis, axis) + \
            ((1 - c) / angle) * cls.wedge(axis)

    @classmethod
    def inv_left_jacobian(cls, phi):
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dim) - 0.5 * cls.wedge(phi)

        axis = phi / angle
        half_angle = 0.5 * angle
        cot_half_angle = 1. / np.tan(half_angle)

        return half_angle * cot_half_angle * np.identity(cls.dim) + \
            (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - \
            half_angle * cls.wedge(axis)

    @classmethod
    def exp(cls, phi):
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return cls(np.identity(cls.dim) + cls.wedge(phi))

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return cls(c * np.identity(cls.dim) +
                   (1 - c) * np.outer(axis, axis) +
                   s * cls.wedge(axis))

    def log(self):
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(self.mat) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1., 1.)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.):
            return self.__class__.vee(self.mat - np.identity(3))

        # Otherwise take the matrix logarithm and return the rotation vector
        return self.__class__.vee((0.5 * angle / np.sin(angle)) * (self.mat - self.mat.T))

    def normalize(self):
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        u, s, v = np.linalg.svd(self.mat, full_matrices=False)

        middle = np.identity(self.dim)
        middle[2, 2] = np.linalg.det(u) * np.linalg.det(v)

        self.mat = u.dot(middle).dot(v)

    def adjoint(self):
        return self.mat

    def dot(self, other):
        if isinstance(other, self.__class__):
            # Compound with another rotation
            return self.__class__(np.dot(self.mat, other.mat))
        else:
            other = np.atleast_2d(other)

            # Transform one or more 3-vectors or fail
            if other.shape[1] == self.dim:
                return np.squeeze(np.dot(self.mat, other.T).T)
            else:
                raise ValueError(
                    "Vector must have shape ({},) or (N,{})".format(self.dim, self.dim))

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
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

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


class SE3(base.SpecialEuclideanGroup):
    """Homogeneous transformation matrix in SE(3) using active (alibi) transformations."""
    dim = 4
    dof = 6
    RotationType = SO3

    def __init__(self, rot, trans):
        super().__init__(rot, trans)

    @classmethod
    def is_valid_matrix(cls, mat):
        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[3, :], np.array([0, 0, 0, 1])) and \
            cls.RotationType.is_valid_matrix(mat[0:3, 0:3])

    @classmethod
    def identity(cls):
        return cls.from_matrix(np.identity(cls.dim))

    @classmethod
    def wedge(cls, xi):
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dim, cls.dim])
        Xi[:, 0:3, 0:3] = cls.RotationType.wedge(xi[:, 3:7])
        Xi[:, 0:3, 3] = xi[:, 0:3]
        return np.squeeze(Xi)

    @classmethod
    def vee(cls, Xi):
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = np.empty([Xi.shape[0], cls.dof])
        xi[:, 0:3] = Xi[:, 0:3, 3]
        xi[:, 3:6] = cls.RotationType.vee(Xi[:, 0:3, 0:3])
        return np.squeeze(xi)

    @classmethod
    def exp(cls, xi):
        if len(xi) != cls.dof:
            raise ValueError("xi must have length 6")

        rho = xi[0:3]
        phi = xi[3:6]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    def log(self):
        phi = self.RotationType.log(self.rot)
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def as_matrix(self):
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (3, 1))
        return np.vstack([np.hstack([R, t]),
                          np.array([0, 0, 0, 1])])

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        rotmat = self.rot.as_matrix()
        return np.vstack(
            [np.hstack([rotmat,
                        self.RotationType.wedge(self.trans).dot(rotmat)]),
             np.hstack([np.zeros((3, 3)), rotmat])]
        )

    def dot(self, other):
        if isinstance(other, self.__class__):
            # Compound with another transformation
            return SE3(self.rot.dot(other.rot),
                       self.rot.dot(other.trans) + self.trans)
        else:
            other = np.atleast_2d(other)

            if other.shape[1] == self.dim - 1:
                # Transform one or more 3-vectors
                return np.squeeze(self.rot.dot(other) + self.trans)
            elif other.shape[1] == self.dim:
                # Transform one or more 4-vectors
                return np.squeeze(self.as_matrix().dot(other.T)).T
            else:
                raise ValueError("Vector must have shape ({},), ({},), (N,{}) or (N,{})".format(
                    self.dim - 1, self.dim, self.dim - 1, self.dim))

    @classmethod
    def odot(cls, vec, directional=False):
        """SE(3) \odot operator as defined by Barfoot."""
        vec = np.atleast_2d(vec)
        result = np.zeros([vec.shape[0], vec.shape[1], cls.dof])

        if vec.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless vec is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, 0:3, 0:3] = np.eye(3)

            result[:, 0:3, 3:6] = cls.RotationType.wedge(-vec)

        elif vec.shape[1] == cls.dim:
            # Broadcast magic
            result[:, 0:3, 0:3] = vec[:, 3][:, None, None] * np.eye(3)
            result[:, 0:3, 3:6] = cls.RotationType.wedge(-vec[:, 0:3])

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)
