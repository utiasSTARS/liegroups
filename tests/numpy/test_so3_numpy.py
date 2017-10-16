import copy

import numpy as np

from liegroups.numpy import SO3


def test_identity():
    T = SO3.identity()
    assert isinstance(T, SO3)


def test_from_rpy_to_rpy():
    r, p, y = np.pi / 4., np.pi / 3., np.pi
    test_r, test_p, test_y = SO3.from_rpy(r, p, y).to_rpy()
    assert np.isclose(test_r, r)
    assert np.isclose(test_p, p)
    assert np.isclose(test_y, y)


def test_dot():
    C = np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]])
    C2 = C.dot(C)
    assert np.allclose((SO3(C).dot(SO3(C))).mat, C2)


def test_rotx():
    C_got = SO3.rotx(np.pi / 2)
    C_expected = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]])
    assert np.allclose(C_got.mat, C_expected)


def test_roty():
    C_got = SO3.roty(np.pi / 2)
    C_expected = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 0]])
    assert np.allclose(C_got.mat, C_expected)


def test_rotz():
    C_got = SO3.rotz(np.pi / 2)
    C_expected = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
    assert np.allclose(C_got.mat, C_expected)


def test_rpy():
    r = np.pi / 12
    p = np.pi / 6
    y = np.pi / 3
    C_got = SO3.from_rpy(r, p, y)
    C_expected = SO3.rotz(y).dot(SO3.roty(p).dot(SO3.rotx(r)))
    assert np.allclose(C_got.mat, C_expected.mat)


def test_quaternion():
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0, 1, 0, 0])
    q3 = np.array([0, 0, 1, 0])
    q4 = np.array([0, 0, 0, 1])
    q5 = 0.5 * np.ones(4)
    q6 = -q5

    assert np.allclose(SO3.from_quaternion(q1).to_quaternion(), q1)
    assert np.allclose(SO3.from_quaternion(q2).to_quaternion(), q2)
    assert np.allclose(SO3.from_quaternion(q3).to_quaternion(), q3)
    assert np.allclose(SO3.from_quaternion(q4).to_quaternion(), q4)
    assert np.allclose(SO3.from_quaternion(q5).to_quaternion(), q5)
    assert np.allclose(SO3.from_quaternion(q5).mat,
                       SO3.from_quaternion(q6).mat)


def test_wedge():
    phi = [1, 2, 3]
    Phi = SO3.wedge(phi)
    phis = np.array([[1, 2, 3], [4, 5, 6]])
    Phis = SO3.wedge(phis)
    assert np.array_equal(Phi, -Phi.T)
    assert np.array_equal(Phis[0, :, :], SO3.wedge(phis[0]))
    assert np.array_equal(Phis[1, :, :], SO3.wedge(phis[1]))


def test_wedge_vee():
    phi = [1, 2, 3]
    Phi = SO3.wedge(phi)
    phis = np.array([[1, 2, 3], [4, 5, 6]])
    Phis = SO3.wedge(phis)
    assert np.array_equal(phi, SO3.vee(Phi))
    assert np.array_equal(phis, SO3.vee(Phis))


def test_left_jacobians():
    phi_small = [0., 0., 0.]
    phi_big = [np.pi / 2, np.pi / 3, np.pi / 4]

    left_jacobian_small = SO3.left_jacobian(phi_small)
    inv_left_jacobian_small = SO3.inv_left_jacobian(phi_small)
    assert np.allclose(left_jacobian_small.dot(inv_left_jacobian_small),
                       np.identity(3))

    left_jacobian_big = SO3.left_jacobian(phi_big)
    inv_left_jacobian_big = SO3.inv_left_jacobian(phi_big)
    assert np.allclose(left_jacobian_big.dot(inv_left_jacobian_big),
                       np.identity(3))


def test_exp_log():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose(SO3.exp(SO3.log(C)).mat, C.mat)


def test_exp_log_zeros():
    C = SO3.exp(np.zeros(3))
    assert np.allclose(SO3.exp(SO3.log(C)).mat, C.mat)


def test_perturb():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    C_copy = copy.deepcopy(C)
    phi = np.array([0.3, 0.2, 0.1])
    C.perturb(phi)
    assert np.allclose(C.as_matrix(), (SO3.exp(phi).dot(C_copy)).as_matrix())


def test_normalize():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    C.mat += 0.1
    C.normalize()
    assert SO3.is_valid_matrix(C.mat)


def test_inv():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose((C.dot(C.inv())).mat, np.identity(3))


def test_adjoint():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose(C.adjoint(), C.mat)


def test_transform_vectorized():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    pt1 = np.array([1, 2, 3])
    pt2 = np.array([4, 5, 3])
    pts = np.array([pt1, pt2])  # 2x3
    Cpt1 = C.dot(pt1)
    Cpt2 = C.dot(pt2)
    Cpts = C.dot(pts)
    assert(
        np.allclose(Cpt1, Cpts[0])
        and np.allclose(Cpt2, Cpts[1])
    )
