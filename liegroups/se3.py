import numpy as np

from liegroups import SO3


class SE3:
    """Homogeneous transformation matrix in SE(3)"""

    dim = 4
    """Dimension of the transformation matrix."""

    dof = 6
    """Underlying degrees of freedom (i.e., dim of the tangent space)."""

    def __init__(self, rot=SO3.identity(), trans=np.zeros(dim - 1)):
        """Create a SE3 object from a translation and a
         rotation."""
        if not isinstance(rot, SO3):
            raise ValueError("rot must be SO3")

        if len(trans) != self.dim - 1:
            raise ValueError("trans must have length 3")

        self.rot = rot
        """Storage for the rotation matrix"""

        self.trans = trans
        """Storage for the translation vector"""

    @classmethod
    def from_matrix(cls, mat):
        """Create a SE3 object from a 4x4 transformation matrix."""
        if not SE3.is_valid_matrix(mat):
            raise ValueError("Invalid transformation matrix")

        return cls(SO3(mat[0:3, 0:3]), mat[0:3, 3])

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[3, :], np.array([0, 0, 0, 1])) and \
            SO3.is_valid_matrix(mat[0:3, 0:3])

    @classmethod
    def identity(cls):
        """Return the identity element."""
        return cls.from_matrix(np.identity(cls.dim))

    @classmethod
    def wedge(cls, xi):
        """SE(3) wedge operator as defined by Barfoot.

        This is the inverse operation to SE3.vee.
        """
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dim, cls.dim])
        Xi[:, 0:3, 0:3] = SO3.wedge(xi[:, 3:7])
        Xi[:, 0:3, 3] = xi[:, 0:3]
        return np.squeeze(Xi)

    @classmethod
    def vee(cls, Xi):
        """SE(3) vee operator as defined by Barfoot.

        This is the inverse operation to SE3.wedge.
        """
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dim, cls.dim, cls.dim, cls.dim))

        xi = np.empty([Xi.shape[0], cls.dof])
        xi[:, 0:3] = Xi[:, 0:3, 3]
        xi[:, 3:6] = SO3.vee(Xi[:, 0:3, 0:3])
        return np.squeeze(xi)

    @classmethod
    def odot(cls, p, **kwargs):
        """SE(3) \odot operator as defined by Barfoot."""
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless otherwise p is a direction
            # vector, in which case the scale is 0
            scale_is_zero = kwargs.get('directional', False)
            if not scale_is_zero:
                result[:, 0:3, 0:3] = np.eye(3)

            result[:, 0:3, 3:6] = -SO3.wedge(p)

        elif p.shape[1] == cls.dim:
            result[:, 0:3, 0:3] = p[:, 3] * np.eye(3)
            result[:, 0:3, 3:6] = -SO3.wedge(p[:, 0:3])

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)

    @classmethod
    def exp(cls, xi):
        """Exponential map for SE(3).

        Computes a transformation matrix from SE(3)
        tangent vector.

        This is the inverse operation to SE3.log.
        """
        if len(xi) != cls.dof:
            raise ValueError("xi must have length 6")

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

    def perturb(self, xi):
        """Perturb the transformation on the left
        by a vector in its local tangent space."""
        perturbed = SE3.exp(xi) * self
        self.rot = perturbed.rot
        self.trans = perturbed.trans

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
        else:
            other = np.atleast_2d(other)

            if other.shape[1] == self.dim - 1:
                # Transform one or more 3-vectors
                return np.squeeze(self.rot * other.T + np.atleast_2d(self.trans).T).T
            elif other.shape[1] == self.dim:
                # Transform one or more 4-vectors
                return np.squeeze(self.as_matrix().dot(other.T)).T
            else:
                raise ValueError("Vector must have shape ({},), ({},), (N,{}) or (N,{})".format(
                    self.dim - 1, self.dim, self.dim - 1, self.dim))

    def __repr__(self):
        return "SE3({})".format(self.as_matrix())
