"""See 'Dual Quaternions' by Yan-Bin Jia for details.
"""
import numpy as np
from liegroups.numpy.so3 import SO3
from liegroups.numpy.quaternion import quat_mult, quat_identity, quat_inv, quat_conj, matrix_to_quaternion


def matrix_to_dual_quaternion(T, ordering='wxyz'):
    """Convert a transformation matrix to a dual quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    p = matrix_to_quaternion(T[0:3, 0:3], ordering)
    dx, dy, dz = 0.5 * T[0:3, 3]
    # Check ordering
    if ordering is 'xyzw':
        q = np.array([dx, dy, dz, 0.0])
    elif ordering is 'wxyz':
        q = np.array([0.0, dx, dy, dz])
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    q = quat_mult(q, p, ordering)
    return np.append(p, q)


def dual_quaternion_to_matrix(d, ordering='wxyz'):
    """Form a transformation matrix from a dual quaternion.

    Valid orderings are 'xyzw' and 'wxyz'.
    """
    p = d[0:4]
    t = 2. * quat_mult(d[4:], quat_inv(p, ordering), ordering)
    if ordering is 'xyzw':
        t = t[0:3]
    elif ordering is 'wxyz':
        t = t[1:]
    # Form the transformation matrix matrix
    R = SO3.from_quaternion(p, ordering=ordering).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def dual_quat_mult(d1, d2, ordering='wxyz'):
    """Product of two dual quaternions.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    p1 = d1[0:4]
    q1 = d1[4:]
    p2 = d2[0:4]
    q2 = d2[4:]
    p = quat_mult(p1, p2, ordering)
    q = quat_mult(p1, q2, ordering) + quat_mult(q1, p2, ordering)
    return np.append(p, q)


def dual_quat_identity(ordering='wxyz'):
    """Dual quaternion representing infinity.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    return np.append(quat_identity(ordering), 4*[0.])


def dual_quat_inv(d, ordering='wxyz'):
    """Inverse of a dual quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    p = d[0:4]
    q = d[4:]
    p_inv = quat_inv(p, ordering)
    return quat_mult(p_inv, dual_quat_identity(ordering) - quat_mult(q, p_inv, ordering), ordering)


if __name__ == '__main__':
    pass
