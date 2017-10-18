import numpy as np

from . import base
from .so3 import SO3


class SE3(base.SpecialEuclideanBase):
    """Homogeneous transformation matrix in SE(3) using active (alibi) transformations.

    .. math::
        SE(3) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~ \\mathbf{C} \\in SO(3), \\mathbf{r} \\in \\mathbb{R}^3 \\right\\} \\\\
        \\mathfrak{se}(3) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\rho} \\\\ \\boldsymbol{\\phi}
            \\end{bmatrix} \\in \\mathbb{R}^6, \\boldsymbol{\\rho} \\in \\mathbb{R}^3, \\boldsymbol{\\phi} \in \\mathbb{R}^3 \\right\\}

    :cvar ~liegroups.SE2.dim: Dimension of the rotation matrix.
    :cvar ~liegroups.SE2.dof: Underlying degrees of freedom (i.e., dimension of the tangent space).
    :ivar rot: Storage for the rotation matrix :math:`\mathbf{C}`.
    :ivar trans: Storage for the translation vector :math:`\mathbf{r}`.
    """
    dim = 4
    """Dimension of the transformation matrix."""
    dof = 6
    """Underlying degrees of freedom (i.e., dimension of the tangent space)."""
    RotationType = SO3

    def __init__(self, rot, trans):
        """Create a transformation from a rotation matrix(unsafe, but faster)."""
        super().__init__(rot, trans)

    @classmethod
    def wedge(cls, xi):
        """SE(3) wedge operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\Xi} =
            \\boldsymbol{\\xi} ^\\wedge =
            \\begin{bmatrix}
                \\boldsymbol{\\phi} ^\\wedge & \\boldsymbol{\\rho} \\\\
                \\mathbf{0} ^ T & 0
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE2.vee`.
        """
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
        """SE(3) vee operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\xi} = \\boldsymbol{\\Xi} ^\\vee

        This is the inverse operation to :meth:`~liegroups.SE3.wedge`.
        """
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
    def left_jacobian(cls, xi):
        """SE(3) left Jacobian.

        .. math::
            \\mathcal{J}(\\boldsymbol{\\xi})
        """
        raise NotImplementedError

    @classmethod
    def inv_left_jacobian(cls, xi):
        """SE(3) inverse left Jacobian.

        .. math::
            \\mathcal{J}^{-1}(\\boldsymbol{\\xi})
        """
        raise NotImplementedError

    @classmethod
    def exp(cls, xi):
        """Exponential map for SE(3), which computes a transformation from a tangent vector:

        .. math::
            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\boldsymbol{\\phi}^\\wedge) & \\mathbf{J} \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE3.log`.
        """
        if len(xi) != cls.dof:
            raise ValueError("xi must have length {}".format(cls.dof))

        rho = xi[0:3]
        phi = xi[3:6]
        return cls(cls.RotationType.exp(phi),
                   cls.RotationType.left_jacobian(phi).dot(rho))

    def log(self):
        """Logarithmic map for SE(3), which computes a tangent vector from a transformation:

        .. math::
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE3.log`.
        """
        phi = self.RotationType.log(self.rot)
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    def adjoint(self):
        """Adjoint matrix of the transformation.

        .. math::
            \\mathbf{\\mathcal{T}} = 
            \\text{Ad}(\\mathbf{T}) = 
            \\begin{bmatrix}
                \\mathbf{C} & \\mathbf{r}^\\wedge\\mathbf{C} \\\\
                \\mathbf{0} & \\mathbf{C}
            \\end{bmatrix}
            \\in \\mathbb{R}^{6 \\times 6}
        """
        rotmat = self.rot.as_matrix()
        return np.vstack(
            [np.hstack([rotmat,
                        self.RotationType.wedge(self.trans).dot(rotmat)]),
             np.hstack([np.zeros((3, 3)), rotmat])]
        )

    @classmethod
    def odot(cls, p, directional=False):
        """SE(3) odot operator as defined by Barfoot. 

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

        If :math:`\\mathbf{p}` is given in Euclidean coordinates and directional=False, the missing scale value :math:`\\eta` is assumed to be 1 and the Jacobian is 3x6. If directional=True, :math:`\\eta` is assumed to be 0:

        .. math::
            \\mathbf{p}^\\odot =
            \\begin{bmatrix}
                \\eta \\mathbf{1} & -\\boldsymbol{\\epsilon}^\\wedge
            \\end{bmatrix}

        If :math:`\\mathbf{p}` is given in Homogeneous coordinates, the Jacobian is 4x6:

        .. math::
            \\mathbf{p}^\\odot =
            \\begin{bmatrix}
                \\eta \\mathbf{1} & -\\boldsymbol{\\epsilon}^\\wedge \\\\
                \\mathbf{0}^T & \\mathbf{0}^T
            \\end{bmatrix}
        """
        p = np.atleast_2d(p)
        result = np.zeros([p.shape[0], p.shape[1], cls.dof])

        if p.shape[1] == cls.dim - 1:
            # Assume scale parameter is 1 unless p is a direction
            # ptor, in which case the scale is 0
            if not directional:
                result[:, 0:3, 0:3] = np.eye(3)

            result[:, 0:3, 3:6] = cls.RotationType.wedge(-p)

        elif p.shape[1] == cls.dim:
            # Broadcast magic
            result[:, 0:3, 0:3] = p[:, 3][:, None, None] * np.eye(3)
            result[:, 0:3, 3:6] = cls.RotationType.wedge(-p[:, 0:3])

        else:
            raise ValueError("p must have shape ({},), ({},), (N,{}) or (N,{})".format(
                cls.dim - 1, cls.dim, cls.dim - 1, cls.dim))

        return np.squeeze(result)
