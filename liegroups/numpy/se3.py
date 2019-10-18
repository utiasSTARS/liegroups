import numpy as np

from liegroups.numpy import _base
from liegroups.numpy.so3 import SO3


class SE3(_base.SpecialEuclideanBase):
    """Homogeneous transformation matrix in :math:`SE(3)` using active (alibi) transformations.

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

    def adjoint(self):
        """Adjoint matrix of the transformation.

        .. math::
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
    def curlyvee(cls, Psi):
        """:math:`SE(3)` curlyvee operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\xi} = 
            \\boldsymbol{\\Psi}^\\curlyvee

        This is the inverse operation to :meth:`~liegroups.SE3.curlywedge`.
        """
        if Psi.ndim < 3:
            Psi = np.expand_dims(Psi, axis=0)

        if Psi.shape[1:3] != (cls.dof, cls.dof):
            raise ValueError("Psi must have shape ({},{}) or (N,{},{})".format(
                cls.dof, cls.dof, cls.dof, cls.dof))

        xi = np.empty([Psi.shape[0], cls.dof])
        xi[:, 0:3] = cls.RotationType.vee(Psi[:, 0:3, 3:6])
        xi[:, 3:6] = cls.RotationType.vee(Psi[:, 0:3, 0:3])

        return np.squeeze(xi)

    @classmethod
    def curlywedge(cls, xi):
        """:math:`SE(3)` curlywedge operator as defined by Barfoot.

        .. math::
            \\boldsymbol{\\Psi} = 
            \\boldsymbol{\\xi}^\\curlywedge = 
            \\begin{bmatrix}
                \\boldsymbol{\\phi}^\\wedge & \\boldsymbol{\\rho}^\\wedge \\\\
                \\mathbf{0} & \\boldsymbol{\\phi}^\\wedge
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE3.curlyvee`.
        """
        xi = np.atleast_2d(xi)
        if xi.shape[1] != cls.dof:
            raise ValueError(
                "xi must have shape ({},) or (N,{})".format(cls.dof, cls.dof))

        Psi = np.zeros([xi.shape[0], cls.dof, cls.dof])
        Psi[:, 0:3, 0:3] = cls.RotationType.wedge(xi[:, 3:6])
        Psi[:, 0:3, 3:6] = cls.RotationType.wedge(xi[:, 0:3])
        Psi[:, 3:6, 3:6] = Psi[:, 0:3, 0:3]

        return np.squeeze(Psi)

    @classmethod
    def exp(cls, xi):
        """Exponential map for :math:`SE(3)`, which computes a transformation from a tangent vector:

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

    @classmethod
    def from_dual_quaternion(cls, dual_quat, ordering='wxyz'):
        """Form a transformation matrix from a dual quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.

        .. math::
            \\mathbf{C} =
            \\begin{bmatrix}
                1 - 2 (y^2 + z^2) & 2 (xy - wz) & 2 (wy + xz) & 2d_x\\\\
                2 (wz + xy) & 1 - 2 (x^2 + z^2) & 2 (yz - wx) & 2d_y\\\\
                2 (xz - wy) & 2 (wx + yz) & 1 - 2 (x^2 + y^2) & 2d_z\\\\
                0 & 0 & 0 & 1
            \\end{bmatrix}
        """
        if ordering is 'xyzw':
            dx, dy, dz, dw = dual_quat[4:]
        elif ordering is 'wxyz':
            dw, dx, dy, dz = dual_quat[4:]
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
        # Form the transformation matrix matrix
        t = 2.*np.array([dx, dy, dz])
        R = SO3.from_quaternion(dual_quat[0:4], ordering=ordering)
        return cls(R, t)

    @classmethod
    def from_study_parameters(cls, study, ordering='wxyz'):
        """Form a transformation matrix from Study parameters.
        Assumes 'wxyz' ordering and that this is not the 'exceptional generator' (i.e. at least 1 of the four rotation
        elements is nonzero).
        """
        delta = sum(study[0:4]**2)
        if delta <= 0.:
            raise ValueError("First four elements cannot be uniformly zero.")
        if ordering=='wxyz':
            x0, x1, x2, x3 = study[0:4]
            y0, y1, y2, y3 = study[4:]
        else:
            raise ValueError(
                "Valid ordering is 'wxyz'. Got '{}'.".format(ordering))
        study_quadric_constraint = x0*y0 + x1*y1 + x2*y2 + x3*y3
        if not np.isclose(study_quadric_constraint, 0.):
            raise ValueError("Study quadric constraint must be satsified")
        q = study[0:4]/np.sqrt(delta)
        R = SO3.from_quaternion(q, ordering=ordering).as_matrix()
        p = -x0*y1 + x1*y0 - x2*y3 + x3*y2
        q = -x0*y2 + x1*y3 + x2*y0 - x3*y1
        r = -x0*y3 - x1*y2 + x2*y1 + x3*y0
        t = 2.*np.array([p, q, r])/delta
        return cls(R, t)

    @classmethod
    def left_jacobian_Q_matrix(cls, xi):
        """The :math:`\\mathbf{Q}` matrix used to compute :math:`\\mathcal{J}` in :meth:`~liegroups.SE3.left_jacobian` and :math:`\\mathcal{J}^{-1}` in :meth:`~liegroups.SE3.inv_left_jacobian`.

        .. math::
            \\mathbf{Q}(\\boldsymbol{\\xi}) =
            \\frac{1}{2}\\boldsymbol{\\rho}^\\wedge &+ 
            \\left( \\frac{\\phi - \\sin \\phi}{\\phi^3} \\right)
                \\left( 
                    \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge + 
                    \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge + 
                    \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge
                \\right) \\\\ &+
            \\left( \\frac{\\phi^2 + 2 \\cos \\phi - 2}{2 \\phi^4} \\right)
                \\left( 
                    \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge + 
                    \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\phi}^\\wedge - 
                    3 \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge
                \\right) \\\\ &+
            \\left( \\frac{2 \\phi - 3 \\sin \\phi + \\phi \\cos \\phi}{2 \\phi^5} \\right)
                \\left( 
                    \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\phi}^\\wedge + 
                    \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\phi}^\\wedge \\boldsymbol{\\rho}^\\wedge \\boldsymbol{\\phi}^\\wedge
                \\right)
        """
        if len(xi) != cls.dof:
            raise ValueError("xi must have length {}".format(cls.dof))

        rho = xi[0:3]  # translation part
        phi = xi[3:6]  # rotation part

        rx = SO3.wedge(rho)
        px = SO3.wedge(phi)

        ph = np.linalg.norm(phi)
        ph2 = ph * ph
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph

        cph = np.cos(ph)
        sph = np.sin(ph)

        m1 = 0.5
        m2 = (ph - sph) / ph3
        m3 = (0.5 * ph2 + cph - 1.) / ph4
        m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

        t1 = rx
        t2 = px.dot(rx) + rx.dot(px) + px.dot(rx).dot(px)
        t3 = px.dot(px).dot(rx) + rx.dot(px).dot(px) - 3. * px.dot(rx).dot(px)
        t4 = px.dot(rx).dot(px).dot(px) + px.dot(px).dot(rx).dot(px)

        return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

    @classmethod
    def inv_left_jacobian(cls, xi):
        """:math:`SE(3)` inverse left Jacobian.

        .. math::
            \\mathcal{J}^{-1}(\\boldsymbol{\\xi}) = 
            \\begin{bmatrix}
                \\mathbf{J}^{-1} & -\\mathbf{J}^{-1} \\mathbf{Q} \\mathbf{J}^{-1} \\\\
                \\mathbf{0} & \\mathbf{J}^{-1}
            \\end{bmatrix}

        with :math:`\\mathbf{J}^{-1}` as in :meth:`liegroups.SO3.inv_left_jacobian` and :math:`\\mathbf{Q}` as in :meth:`~liegroups.SE3.left_jacobian_Q_matrix`.
        """
        rho = xi[0:3]  # translation part
        phi = xi[3:6]  # rotation part

        # Near |phi|==0, use first order Taylor expansion
        if np.isclose(np.linalg.norm(phi), 0.):
            return np.identity(cls.dof) - 0.5 * cls.curlywedge(xi)

        so3_inv_jac = SO3.inv_left_jacobian(phi)
        Q_mat = cls.left_jacobian_Q_matrix(xi)

        jac = np.zeros([cls.dof, cls.dof])
        jac[0:3, 0:3] = so3_inv_jac
        jac[0:3, 3:6] = -so3_inv_jac.dot(Q_mat).dot(so3_inv_jac)
        jac[3:6, 3:6] = so3_inv_jac

        return jac

    @classmethod
    def left_jacobian(cls, xi):
        """:math:`SE(3)` left Jacobian.

        .. math::
            \\mathcal{J}(\\boldsymbol{\\xi}) = 
            \\begin{bmatrix}
                \\mathbf{J} & \\mathbf{Q} \\\\
                \\mathbf{0} & \\mathbf{J}
            \\end{bmatrix}

        with :math:`\\mathbf{J}` as in :meth:`liegroups.SO3.left_jacobian` and :math:`\\mathbf{Q}` as in :meth:`~liegroups.SE3.left_jacobian_Q_matrix`.
        """
        rho = xi[0:3]  # translation part
        phi = xi[3:6]  # rotation part

        # Near |phi|==0, use first order Taylor expansion
        if np.isclose(np.linalg.norm(phi), 0.):
            return np.identity(cls.dof) + 0.5 * cls.curlywedge(xi)

        so3_jac = SO3.left_jacobian(phi)
        Q_mat = cls.left_jacobian_Q_matrix(xi)

        jac = np.zeros([cls.dof, cls.dof])
        jac[0:3, 0:3] = so3_jac
        jac[0:3, 3:6] = Q_mat
        jac[3:6, 3:6] = so3_jac

        return jac

    def log(self):
        """Logarithmic map for :math:`SE(3)`, which computes a tangent vector from a transformation:

        .. math::
            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}

        This is the inverse operation to :meth:`~liegroups.SE3.exp`.
        """
        phi = self.RotationType.log(self.rot)
        rho = self.RotationType.inv_left_jacobian(phi).dot(self.trans)
        return np.hstack([rho, phi])

    @classmethod
    def odot(cls, p, directional=False):
        """:math:`SE(3)` odot operator as defined by Barfoot.

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

    @classmethod
    def quat_mult(cls, q1, q2, ordering='wxyz'):
        """Multiply two quaternions.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        if ordering is 'xyzw':
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
        elif ordering is 'wxyz':
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
        else:
            raise ValueError(
                "Valid ordering is 'wxyz'. Got '{}'.".format(ordering))



    def to_dual_quaternion(self, ordering='wxyz'):
        """Convert a transformation matrix to a dual quaternion.

           Valid orderings are 'xyzw' and 'wxyz'.
        """
        so3_obj = SO3(self.rot.as_matrix())
        q = so3_obj.to_quaternion(ordering)
        dx, dy, dz = 0.5*self.trans
        # Check ordering
        if ordering is 'xyzw':
            if np.abs(q[3]) > 0:
                dw = -np.dot(q[0:3], np.array([dx, dy, dz])) / q[3]
                dual_part = np.array([dx, dy, dz, dw])
            else:
                raise ValueError(
                    "Quaternion real part must be non-zero (i.e. no rotations by pi rad)")
        elif ordering is 'wxyz':
            if np.abs(q[0]) > 0:
                dw = -np.dot(q[1:4], np.array([dx, dy, dz])) / q[0]
                dual_part = np.array([dw, dx, dy, dz])
            else:
                raise ValueError(
                    "Quaternion real part must be non-zero (i.e. no rotations by pi rad)")
        else:
            raise ValueError(
                "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
        dual_quat = np.append(q, dual_part)
        return dual_quat

    @classmethod
    def vee(cls, Xi):
        """:math:`SE(3)` vee operator as defined by Barfoot.

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
    def wedge(cls, xi):
        """:math:`SE(3)` wedge operator as defined by Barfoot.

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
        Xi[:, 0:3, 0:3] = cls.RotationType.wedge(xi[:, 3:6])
        Xi[:, 0:3, 3] = xi[:, 0:3]
        return np.squeeze(Xi)
