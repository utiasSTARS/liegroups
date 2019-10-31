import numpy as np

from . import _base
from .so2 import SO2Matrix


class SE2Matrix(_base.SEMatrixBase):
    """Homogeneous transformation matrix in :math:`SE(2)` using active (alibi) transformations.

    .. math::
        SE(2) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{3 \\times 3} ~\\middle|~ \\mathbf{C} \\in SO(2), \\mathbf{r} \\in \\mathbb{R}^2 \\right\\} \\\\
        \\mathfrak{se}(2) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{3 \\times 3} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\rho} \\\\ \\phi
            \\end{bmatrix} \\in \\mathbb{R}^3, \\boldsymbol{\\rho} \\in \\mathbb{R}^2, \\phi \\in \\mathbb{R} \\right\\}

    :cvar ~liegroups.SE2.dim: Dimension of the rotation matrix.
    :cvar ~liegroups.SE2.dof: Underlying degrees of freedom (i.e., dimension of the tangent space).
    :ivar rot: Storage for the rotation matrix :math:`\\mathbf{C}`.
    :ivar trans: Storage for the translation vector :math:`\\mathbf{r}`.
    """
    dim = 3
    """Dimension of the transformation matrix."""
    dof = 3
    """Underlying degrees of freedom (i.e., dimension of the tangent space)."""
    RotationType = SO2Matrix

    def adjoint(self):
        """Adjoint matrix of the transformation.

        .. math::
            \\text{Ad}(\\mathbf{T}) = 
            \\begin{bmatrix}
                \\mathbf{C} & 1^\\wedge \\mathbf{r} \\\\
                \\mathbf{0}^T & 1
            \\end{bmatrix}
            \\in \\mathbb{R}^{3 \\times 3}
        """
        rot_part = self.rot.as_matrix()
        trans_part = np.array([self.trans[1], -self.trans[0]]).reshape((2, 1))
        return np.vstack([np.hstack([rot_part, trans_part]),
                          [0, 0, 1]])

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE(2)`, which computes a transformation from a tangent vector:

        .. math::
            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\phi ^\\wedge) & \\mathbf{J} \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE2.log`.
        """
        if len(xi) != cls.dof:
            raise ValueError("xi must have length {}".format(cls.dof))

        rho = xi[0:2]
        phi = xi[2]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    @classmethod
    def inv_left_jacobian(cls, xi):
        """:math:`SE(2)` inverse left Jacobian.

        .. math::
            \\mathcal{J}^{-1}(\\boldsymbol{\\xi})
        """
        raise NotImplementedError

    @classmethod
    def left_jacobian(cls, xi):
        """:math:`SE(2)` left Jacobian.

        .. math::
            \\mathcal{J}(\\boldsymbol{\\xi})
        """
        raise NotImplementedError

    def log(self):
        """Logarithmic map for :math:`SE(2)`, which computes a tangent vector from a transformation:

        .. math::
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE2.log`.
        """
        phi = self.rot.log()
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    @classmethod
    def odot(cls, p, directional=False):
        """:math:`SE(2)` odot operator as defined by Barfoot. 

        This is the Jacobian of a vector 

        .. math::
            \\mathbf{p} = 
            \\begin{bmatrix}
                sx \\\\ sy \\\\ sz \\\\ s
            \\end{bmatrix} =
            \\begin{bmatrix}
                \\boldsymbol{\\epsilon} \\\\ \\eta
            \\end{bmatrix}

        with respect to a perturbation in the underlying parameters of :math:`\\mathbf{T}`.

        If :math:`\\mathbf{p}` is given in Euclidean coordinates and directional=False, the missing scale value :math:`\\eta` is assumed to be 1 and the Jacobian is 2x3. If directional=True, :math:`\\eta` is assumed to be 0:

        .. math::
            \\mathbf{p}^\\odot =
            \\begin{bmatrix}
                \\eta \\mathbf{1} & 1^\\wedge \\boldsymbol{\\epsilon}
            \\end{bmatrix}

        If :math:`\\mathbf{p}` is given in Homogeneous coordinates, the Jacobian is 3x3:

        .. math::
            \\mathbf{p}^\\odot =
            \\begin{bmatrix}
                \\eta \\mathbf{1} & 1^\\wedge \\boldsymbol{\\epsilon} \\\\
                \\mathbf{0}^T & 0
            \\end{bmatrix}
        """
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # vector, in which case the scale is 0
            if not directional:
                result[:, 0:2, 0:2] = np.eye(2)

            result[:, 0:2, 2] = cls.RotationType.wedge(1).dot(p.T).T

        elif p.shape[1] == cls.dim:
            result[:, 0:2, 0:2] = p[:, 2] * np.eye(2)
            result[:, 0:2, 2] = cls.RotationType.wedge(1).dot(p[:, 0:2].T).T

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)

    @classmethod
    def vee(cls, Xi):
        """:math:`SE(2)` vee operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\xi} = \\boldsymbol{\\Xi} ^\\vee

        This is the inverse operation to :meth:`~liegroups.SE2.wedge`.
        """
        if Xi.ndim < 3:
            Xi = np.expand_dims(Xi, axis=0)

        if Xi.shape[1:3] != (cls.dof, cls.dof):
            raise ValueError("Xi must have shape ({},{}) or (N,{},{})".format(
                cls.dof, cls.dof, cls.dof, cls.dof))

        xi = np.empty([Xi.shape[0], cls.dof])
        xi[:, 0:2] = Xi[:, 0:2, 2]
        xi[:, 2] = cls.RotationType.vee(Xi[:, 0:2, 0:2])
        return np.squeeze(xi)

    @classmethod
    def wedge(cls, xi):
        """:math:`SE(2)` wedge operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\Xi} =
            \\boldsymbol{\\xi} ^\\wedge =
            \\begin{bmatrix}
                \\phi ^\\wedge & \\boldsymbol{\\rho} \\\\
                \\mathbf{0} ^ T & 0
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE2.vee`.
        """
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Xi = np.zeros([xi.shape[0], cls.dof, cls.dof])
        Xi[:, 0:2, 0:2] = cls.RotationType.wedge(xi[:, 2])
        Xi[:, 0:2, 2] = xi[:, 0:2]

        return np.squeeze(Xi)
