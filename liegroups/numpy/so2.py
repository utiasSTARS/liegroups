import numpy as np

from . import _base


class SO2Matrix(_base.SOMatrixBase):
    """Rotation matrix in :math:`SO(2)` using active (alibi) transformations.

    .. math::
        SO(2) &= \\left\\{ \\mathbf{C} \\in \\mathbb{R}^{2 \\times 2} ~\\middle|~ \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det \\mathbf{C} = 1 \\right\\} \\\\
        \\mathfrak{so}(2) &= \\left\\{ \\boldsymbol{\\Phi} = \\phi^\\wedge \\in \\mathbb{R}^{2 \\times 2} ~\\middle|~ \\phi \\in \\mathbb{R} \\right\\}

    :cvar ~liegroups.SO2.dim: Dimension of the rotation matrix.
    :cvar ~liegroups.SO2.dof: Underlying degrees of freedom (i.e., dimension of the tangent space).
    :ivar mat: Storage for the rotation matrix :math:`\\mathbf{C}`.
    """
    dim = 2
    """Dimension of the transformation matrix."""
    dof = 1
    """Underlying degrees of freedom (i.e., dimension of the tangent space)."""

    def adjoint(self):
        """Adjoint matrix of the transformation.

        .. math::
            \\text{Ad}(\\mathbf{C}) = 1
        """
        return 1.

    @classmethod
    def exp(cls, phi):
        """Exponential map for :math:`SO(2)`, which computes a transformation from a tangent vector:

        .. math::
            \\mathbf{C}(\\phi) = 
            \\exp(\\phi^\\wedge) =
            \\cos \\phi \\mathbf{1} + \\sin \\phi 1^\\wedge = 
            \\begin{bmatrix}
                \\cos \\phi  & -\\sin \\phi  \\\\
                \\sin \\phi & \\cos \\phi
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SO2.log`.
        """
        c = np.cos(phi)
        s = np.sin(phi)

        return cls(np.array([[c, -s],
                             [s,  c]]))

    @classmethod
    def from_angle(cls, angle_in_radians):
        """Form a rotation matrix given an angle in radians.

        See :meth:`~liegroups.SO2.exp`
        """
        return cls.exp(angle_in_radians)

    @classmethod
    def inv_left_jacobian(cls, phi):
        """:math:`SO(2)` inverse left Jacobian.

        .. math::
            \\mathbf{J}^{-1}(\\phi) = 
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\phi^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} -
                \\frac{\\phi}{2} 1^\\wedge, & \\text{otherwise}
            \\end{cases}
        """
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) - 0.5 * cls.wedge(phi)

        half_angle = 0.5 * phi
        cot_half_angle = 1. / np.tan(half_angle)
        return half_angle * cot_half_angle * np.identity(cls.dim) - \
            half_angle * cls.wedge(1.)

    @classmethod
    def left_jacobian(cls, phi):
        """:math:`SO(2)` left Jacobian.

        .. math::
            \\mathbf{J}(\\phi) = 
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\phi^\\wedge, & \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} - 
                \\frac{1 - \\cos \\phi}{\\phi} 1^\\wedge, & \\text{otherwise}
            \\end{cases}
        """
        # Near phi==0, use first order Taylor expansion
        if np.isclose(phi, 0.):
            return np.identity(cls.dim) + 0.5 * cls.wedge(phi)

        s = np.sin(phi)
        c = np.cos(phi)

        return (s / phi) * np.identity(cls.dim) + \
            ((1 - c) / phi) * cls.wedge(1.)

    def log(self):
        """Logarithmic map for :math:`SO(2)`, which computes a tangent vector from a transformation:

        .. math::
            \\phi(\\mathbf{C}) = 
            \\ln(\\mathbf{C})^\\vee =
            \\text{atan2}(C_{1,0}, C_{0,0})

        This is the inverse operation to :meth:`~liegroups.SO2.exp`.
        """
        c = self.mat[0, 0]
        s = self.mat[1, 0]
        return np.arctan2(s, c)

    def to_angle(self):
        """Recover the rotation angle in radians from the rotation matrix.

        See :meth:`~liegroups.SO2.log`
        """
        return self.log()

    @classmethod
    def vee(cls, Phi):
        """:math:`SO(2)` vee operator as defined by Barfoot.

        .. math::
            \\phi = \\boldsymbol{\\Phi}^\\vee

        This is the inverse operation to :meth:`~liegroups.SO2.wedge`.
        """
        if Phi.ndim < 3:
            Phi = np.expand_dims(Phi, axis=0)

        if Phi.shape[1:3] != (cls.dim, cls.dim):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(
                    cls.dim, cls.dim, cls.dim, cls.dim))

        return np.squeeze(Phi[:, 1, 0])

    @classmethod
    def wedge(cls, phi):
        """:math:`SO(2)` wedge operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\Phi} = 
            \\phi^\\wedge = 
            \\begin{bmatrix} 
                0 & -\\phi \\\\
                \\phi & 0 
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SO2.vee`.
        """
        phi = np.atleast_1d(phi)

        Phi = np.zeros([len(phi), cls.dim, cls.dim])
        Phi[:, 0, 1] = -phi
        Phi[:, 1, 0] = phi
        return np.squeeze(Phi)
