import numpy as np

from liegroups import SO2


class SE2:
    """Homogeneous transformation matrix in SE(2) using active (alibi) transformations."""

    dim = 3
    """Dimension of the transformation matrix."""

    dof = 3
    """Underlying degrees of freedom (i.e., dim of the tangent space)."""

    def __init__(self, rot=SO2.identity(), trans=np.zeros(dim - 1)):
        """Create a SE2 object from a translation and a
         rotation."""
        if not isinstance(rot, SO2):
            rot = SO2(rot)

        if len(trans) != self.dim - 1:
            raise ValueError("trans must have length 2")

        self.rot = rot
        """Storage for the rotation matrix"""

        self.trans = trans
        """Storage for the translation vector"""

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a SE2 object from a 3x3 transformation matrix."""
        mat_is_valid = cls.is_valid_matrix(mat)
        if mat_is_valid:
            result = cls(SO2(mat[0:2, 0:2]), mat[0:2, 2])
        elif not mat_is_valid and normalize:
            result = cls(SO2(mat[0:2, 0:2]), mat[0:2, 2])
            result.normalize()
        else:
            raise ValueError(
                "Invalid transformation matrix. Use normalize=True to handle rounding errors.")

        return result

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

    @classmethod
    def exp(cls, xi):
        """Exponential map for SE(2).

        Computes a transformation matrix from SE(2)
        tangent vector.

        This is the inverse operation to SE2.log.
        """
        if len(xi) != cls.dof:
            raise ValueError("xi must have length 3")

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
        perturbed = SE2.exp(xi).dot(self)
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
        inv_trans = -(inv_rot.dot(self.trans))
        return SE2(inv_rot, inv_trans)

    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        rotpart = self.rot.as_matrix()
        transpart = np.array([self.trans[1], -self.trans[0]]).reshape((2, 1))
        return np.vstack([np.hstack([rotpart, transpart]),
                          [0, 0, 1]])

    def dot(self, other):
        """Transform another transformation or one or more vectors.
            The multiplication operator is equivalent to dot.
        """
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

    def __repr__(self):
        return "SE2({})".format(str(self.as_matrix()).replace('\n', '\n    '))
