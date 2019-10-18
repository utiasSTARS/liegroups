import numpy as np

from liegroups.numpy.so3 import SO3
from liegroups.numpy.quaternion import quat_mult, quat_left_mult_matrix, quat_right_mult_matrix, quat_inv, \
                                            quat_identity, quat_rotate, matrix_to_quaternion, quaternion_to_matrix


def test_conversion():
    q = np.random.rand(4) * 2 - 1
    q = q / np.linalg.norm(q)
    R = quaternion_to_matrix(q)
    q2 = matrix_to_quaternion(R)
    assert np.allclose(q, q2) or np.allclose(q, -q2)


def test_quat_mult():
    q1 = np.random.rand(4) * 2 - 1
    q1 = q1 / np.linalg.norm(q1)
    q2 = np.random.rand(4) * 2 - 1
    q2 = q2 / np.linalg.norm(q2)
    q = quat_mult(q1, q2)

    R1 = SO3.from_quaternion(q1).as_matrix()
    R2 = SO3.from_quaternion(q2).as_matrix()
    R = np.dot(R1, R2)
    Rq = SO3.from_quaternion(q).as_matrix()

    assert np.allclose(R, Rq)


def  test_mult_matrices():
    ordering = 'wxyz'
    q1 = np.random.rand(4)*2 - 1
    q1 = q1/np.linalg.norm(q1)
    q2 = np.random.rand(4)*2 - 1
    q2 = q2/np.linalg.norm(q2)
    q = quat_mult(q1, q2, ordering)
    q_l = np.dot(quat_left_mult_matrix(q1, ordering), q2)
    q_r = np.dot(quat_right_mult_matrix(q2, ordering), q1)
    assert np.allclose(q, q_l)
    assert np.allclose(q, q_r)


def test_quat_inv():
    q = np.random.rand(4) * 2 - 1
    q = q / np.linalg.norm(q)
    q_inv = quat_inv(q)
    assert np.allclose(quat_identity(), quat_mult(q, q_inv))
    assert np.allclose(quat_identity(), quat_mult(q_inv, q))


def test_quat_rotate():
    q = np.random.rand(4) * 2 - 1
    q = q / np.linalg.norm(q)
    v = np.random.rand(3)*2 - 1
    R = SO3.from_quaternion(q).as_matrix()
    v_q = quat_rotate(q, v)
    v_R = np.dot(R, v)
    assert np.allclose(v_q, v_R)