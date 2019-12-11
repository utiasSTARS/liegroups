import numpy as np

#from liegroups.numpy.so3 import SO3
from liegroups.numpy.dual_quaternion import matrix_to_dual_quaternion, dual_quaternion_to_matrix, dual_quat_mult, \
                                            dual_quat_inv
from liegroups.numpy.quaternion import quaternion_to_matrix

def test_conversion():
    q = np.random.rand(4) * 2 - 1
    q = q / np.linalg.norm(q)
    R = quaternion_to_matrix(q)
    t = np.random.rand(3)*2 - 1
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    d = matrix_to_dual_quaternion(T)
    T2 = dual_quaternion_to_matrix(d)
    assert np.allclose(T[0:3, 0:3], T2[0:3, 0:3])
    assert np.allclose(T[0:3, 3], T2[0:3, 3])



def test_transformation():
    q = np.random.rand(4) * 2 - 1
    q = q / np.linalg.norm(q)
    R = quaternion_to_matrix(q)
    t = np.random.rand(3)*2 - 1
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    d = matrix_to_dual_quaternion(T)

    q1 = np.random.rand(4)*2 - 1
    q1 = q1 / np.linalg.norm(q1)
    R1 = quaternion_to_matrix(q1)
    t1 = np.random.rand(3) * 2 - 1
    T1 = np.eye(4)
    T1[0:3, 0:3] = R1
    T1[0:3, 3] = t1
    a = matrix_to_dual_quaternion(T1)

    a_out = dual_quat_mult(d, dual_quat_mult(a, dual_quat_inv(d)))
