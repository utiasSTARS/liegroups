import numpy as np
from liegroups.numpy.so3 import SO3


def matrix_to_quaternion(R, ordering='wxyz'):
    """Convert a rotation matrix to a unit quaternion.

    Valid orderings are 'xyzw' and 'wxyz'.
    """
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
    # TODO: make it (4,1) instead of (4,)? Replace with SO3.to_quaternion() call? Or call this in SO3.to_quaternion()?
    return quat


def quaternion_to_matrix(q, ordering='wxyz'):
    """Convert a unit quaternion to a rotation matrix.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    return SO3.from_quaternion(q, ordering).as_matrix()

def quat_conj(q, ordering='wxyz'):
    """Conjugate of a quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    q_out = q.copy()
    if ordering is 'xyzw':
        q_out[0:3] = -q_out[0:3]
    elif ordering is 'wxyz':
        q_out[1:] = -q_out[1:]
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    return q_out


def quat_mult(q1, q2, ordering='wxyz'):
    """Hamilton product of two quaternions.

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
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    if ordering is 'xyzw':
        q = np.array([x, y, z, w])
    elif ordering is 'wxyz':
        q = np.array([w, x, y, z])
    return q


def quat_identity(ordering='wxyz'):
    """Return the identity quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    if ordering is 'xyzw':
        return np.array([0., 0., 0., 1.])
    elif ordering is 'wxyz':
        return np.array([1., 0., 0., 0.])
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))


def quat_inv(q, ordering='wxyz'):
    """Invert a quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    if ordering is 'xyzw':
        q_inv = q.copy()
        q_inv[0:3] = -q[0:3]
    elif ordering is 'wxyz':
        q_inv = q.copy()
        q_inv[1:] = -q[1:]
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    return q_inv


def quat_left_mult_matrix(q, ordering='wxyz'):
    """Left-multiplication matrix of a quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    if ordering is 'xyzw':
        x, y, z, w = q
        i, j, k, l = (0, 1, 2, 3)
    elif ordering is 'wxyz':
        w, x, y, z = q
        i, j, k, l = (1, 2, 3, 0)
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))
    Q_left = np.zeros((4, 4)) * np.nan
    np.fill_diagonal(Q_left, w)

    Q_left[i, j] = -z
    Q_left[i, k] = y
    Q_left[i, l] = x

    Q_left[j, i] = z
    Q_left[j, k] = -x
    Q_left[j, l] = y

    Q_left[k, i] = -y
    Q_left[k, j] = x
    Q_left[k, l] = z

    Q_left[l, i] = -x
    Q_left[l, j] = -y
    Q_left[l, k] = -z

    return Q_left


def quat_right_mult_matrix(q, ordering='wxyz'):
    """Right-multiplication matrix of a quaternion.

       Valid orderings are 'xyzw' and 'wxyz'.
    """
    if ordering is 'xyzw':
        x, y, z, w = q
        i, j, k, l = (0, 1, 2, 3)

    elif ordering is 'wxyz':
        w, x, y, z = q
        i, j, k, l = (1, 2, 3, 0)
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    Q_right = np.zeros((4, 4)) * np.nan
    np.fill_diagonal(Q_right, w)

    Q_right[i, j] = z
    Q_right[i, k] = -y
    Q_right[i, l] = x

    Q_right[j, i] = -z
    Q_right[j, k] = x
    Q_right[j, l] = y

    Q_right[k, i] = y
    Q_right[k, j] = -x
    Q_right[k, l] = z

    Q_right[l, i] = -x
    Q_right[l, j] = -y
    Q_right[l, k] = -z

    return Q_right

def quat_rotate(q, v, ordering='wxyz'):
    """Rotate a 3 dim vector with a quaternion.

        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if ordering is 'xyzw':
        v_pure = np.append(v, 0.)
    elif ordering is 'wxyz':
        v_pure = np.append(0., v)
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    v_rot = quat_mult(quat_mult(q, v_pure, ordering), quat_inv(q, ordering), ordering)

    if ordering is 'xyzw':
        return v_rot[0:3]
    else:
        return v_rot[1:4]

if __name__ == '__main__':
    q = np.random.rand(4)
    q = q/np.linalg.norm(q)

    print(q)
    print(quat_conj(q))
    print(q)