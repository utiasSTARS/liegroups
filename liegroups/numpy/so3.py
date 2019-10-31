import numpy as np

from . import _base


class SO3Matrix(_base.SOMatrixBase):
    """Rotation matrix in :math:`SO(3)` using active (alibi) transformations.

    .. math::
        SO(3) &= \\left\\{ \\mathbf{C} \\in \\mathbb{R}^{3 \\times 3} ~\\middle|~ \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det \\mathbf{C} = 1 \\right\\} \\\\
        \\mathfrak{so}(3) &= \\left\\{ \\boldsymbol{\\Phi} = \\boldsymbol{\\phi}^\\wedge \\in \\mathbb{R}^{3 \\times 3} ~\\middle|~ \\boldsymbol{\\phi} = \\phi \\mathbf{a} \\in \\mathbb{R}^3, \\phi = \\Vert \\boldsymbol{\\phi} \\Vert \\right\\}

    :cvar ~liegroups.SO3.dim: Dimension of the rotation matrix.
    :cvar ~liegroups.SO3.dof: Underlying degrees of freedom (i.e., dimension of the tangent space).
    :ivar mat: Storage for the rotation matrix :math:`\\mathbf{C}`.
    """
    dim = 3
    """Dimension of the transformation matrix."""
    dof = 3
    """Underlying degrees of freedom (i.e., dimension of the tangent space)."""

    def adjoint(self):
        """Adjoint matrix of the transformation.

        .. math::
            \\text{Ad}(\\mathbf{C}) = \\mathbf{C}
            \\in \\mathbb{R}^{3 \\times 3}
        """
        return self.mat

    @classmethod
    def exp(cls, phi):
        """Exponential map for :math:`SO(3)`, which computes a transformation from a tangent vector:

        .. math::
            \\mathbf{C}(\\boldsymbol{\\phi}) =
            \\exp(\\boldsymbol{\\phi}^\\wedge) =
            \\begin{cases}
                \\mathbf{1} + \\boldsymbol{\\phi}^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\cos \\phi \\mathbf{1} +
                (1 - \\cos \\phi) \\mathbf{a}\\mathbf{a}^T +
                \\sin \\phi \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~liegroups.SO3.log`.
        """
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

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.

        .. math::
            \\mathbf{C} = 
            \\begin{bmatrix}
                1 - 2 (y^2 + z^2) & 2 (xy - wz) & 2 (wy + xz) \\\\
                2 (wz + xy) & 1 - 2 (x^2 + z^2) & 2 (yz - wx) \\\\
                2 (xz - wy) & 2 (wx + yz) & 1 - 2 (x^2 + y^2)
            \\end{bmatrix}
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

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles :math:`(\\alpha, \\beta, \\gamma)`.

        .. math::
            \\mathbf{C} = \\mathbf{C}_z(\\gamma) \\mathbf{C}_y(\\beta) \\mathbf{C}_x(\\alpha)
        """
        return cls.rotz(yaw).dot(cls.roty(pitch).dot(cls.rotx(roll)))

    @classmethod
    def inv_left_jacobian(cls, phi):
        """:math:`SO(3)` inverse left Jacobian.

        .. math::
            \\mathbf{J}^{-1}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\boldsymbol{\\phi}^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} +
                \\left( 1 - \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\right) \\mathbf{a}\\mathbf{a}^T -
                \\frac{\\phi}{2} \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}
        """
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dof) - 0.5 * cls.wedge(phi)

        axis = phi / angle
        half_angle = 0.5 * angle
        cot_half_angle = 1. / np.tan(half_angle)

        return half_angle * cot_half_angle * np.identity(cls.dof) + \
            (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - \
            half_angle * cls.wedge(axis)

    @classmethod
    def left_jacobian(cls, phi):
        """:math:`SO(3)` left Jacobian.

        .. math::
            \\mathbf{J}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\boldsymbol{\\phi}^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} +
                \\left(1 - \\frac{\\sin \\phi}{\\phi} \\right) \\mathbf{a}\\mathbf{a}^T +
                \\frac{1 - \\cos \\phi}{\\phi} \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}
        """
        if len(phi) != cls.dof:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near |phi|==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(cls.dof) + 0.5 * cls.wedge(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return (s / angle) * np.identity(cls.dof) + \
            (1 - s / angle) * np.outer(axis, axis) + \
            ((1 - c) / angle) * cls.wedge(axis)

    def log(self):
        """Logarithmic map for :math:`SO(3)`, which computes a tangent vector from a transformation:

        .. math::
            \\phi &= \\frac{1}{2} \\left( \\mathrm{Tr}(\\mathbf{C}) - 1 \\right) \\\\
            \\boldsymbol{\\phi}(\\mathbf{C}) &= 
            \\ln(\\mathbf{C})^\\vee =
            \\begin{cases}
                \\mathbf{1} - \\boldsymbol{\\phi}^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\left( \\frac{1}{2} \\frac{\\phi}{\\sin \\phi} \\left( \\mathbf{C} - \\mathbf{C}^T \\right) \\right)^\\vee, & \\text{otherwise}
            \\end{cases}

        This is the inverse operation to :meth:`~liegroups.SO3.log`.
        """
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(self.mat) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1., 1.)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.):
            return self.vee(self.mat - np.identity(3))

        # Otherwise take the matrix logarithm and return the rotation vector
        return self.vee((0.5 * angle / np.sin(angle)) * (self.mat - self.mat.T))

    @classmethod
    def rotx(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis.

        .. math::
            \\mathbf{C}_x(\\phi) = 
            \\begin{bmatrix}
                1 & 0 & 0 \\\\
                0 & \\cos \\phi & -\\sin \\phi \\\\
                0 & \\sin \\phi & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[1., 0., 0.],
                             [0., c, -s],
                             [0., s,  c]]))

    @classmethod
    def roty(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis.

        .. math::
            \\mathbf{C}_y(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & 0 & \\sin \\phi \\\\
                0 & 1 & 0 \\\\
                \\sin \\phi & 0 & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[c,  0., s],
                             [0., 1., 0.],
                             [-s, 0., c]]))

    @classmethod
    def rotz(cls, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis.

        .. math::
            \\mathbf{C}_z(\\phi) = 
            \\begin{bmatrix}
                \\cos \\phi & -\\sin \\phi & 0 \\\\
                \\sin \\phi  & \\cos \\phi & 0 \\\\
                0 & 0 & 1
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return cls(np.array([[c, -s,  0.],
                             [s,  c,  0.],
                             [0., 0., 1.]]))

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
            qz = (R[1, 0] - R[0, 1]) / d

        # Check ordering last
        if ordering is 'xyzw':
            quat = np.array([qx, qy, qz, qw])
        elif ordering is 'wxyz':
            quat = np.array([qw, qx, qy, qz])
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

        return quat

    def to_rpy(self):
        """Convert a rotation matrix to RPY Euler angles :math:`(\\alpha, \\beta, \\gamma)`."""
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
    def vee(cls, Phi):
        """:math:`SO(3)` vee operator as defined by Barfoot.

        .. math::
            \\phi = \\boldsymbol{\\Phi}^\\vee

        This is the inverse operation to :meth:`~liegroups.SO3.wedge`.
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
    def wedge(cls, phi):
        """:math:`SO(3)` wedge operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\Phi} =
            \\boldsymbol{\\phi}^\\wedge =
            \\begin{bmatrix}
                0 & -\\phi_3 & \\phi_2 \\\\
                \\phi_3 & 0 & -\\phi_1 \\\\
                -\\phi_2 & \\phi_1 & 0
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SO3.vee`.
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


class SO3Quaternion(_base.VectorLieGroupBase):
    """Rotation in SO(3) using unit-length quaternions (wxyz ordering)."""

    dim = 4
    dof = 3

    def from_array(self, arr, ordering='wxyz'):
        if ordering is 'xyzw':
            self.data = arr[[3, 0, 1, 2]]
        elif ordering is 'wxyz':
            self.data = arr
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    def dot(self, other):
        """Multiply another rotation or one or more vectors on the left.
        """
        if isinstance(other, self.__class__):
            # Compound with another rotation
            pv = p[1:]
            qv = q[1:]

            r = np.hstack([p[0]*q[0] - np.dot(pv, qv),
                           p[0]*qv + q[0]*pv + np.dot(skew(pv), qv)])
            return 0
        else:
            other = np.atleast_2d(other)

            # Transform one or more 2-vectors or fail
            if other.shape[1] == self.dim:
                return 0
            else:
                raise ValueError(
                    "Vector must have shape ({},) or (N,{})".format(self.dim, self.dim))

    @classmethod
    def identity(cls):
        """Return the identity rotation."""
        return cls(np.array([1, 0, 0, 0]))

    def inv(self):
        """Return the inverse rotation:

        .. math::
            \\mathbf{C}^{-1} = \\mathbf{C}^T
        """
        inv = self.conjugate()
        inv.data = inv.data / np.dot(inv.data, inv.data)

        return inv
