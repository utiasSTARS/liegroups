import copy

import numpy as np

from liegroups import SO3


def test_identity():
    T = SO3.identity()
    assert isinstance(T, SO3)


def test_from_rpy_to_rpy():
    r, p, y = np.pi / 4., np.pi / 3., np.pi
    test_r, test_p, test_y = SO3.from_rpy(r, p, y).to_rpy()
    assert np.isclose(test_r, r)
    assert np.isclose(test_p, p)
    assert np.isclose(test_y, y)


def test_mul():
    C = np.array([[0, 0, -1],
                  [0, 1, 0],
                  [1, 0, 0]])
    C2 = C.dot(C)
    assert np.allclose((SO3(C) * SO3(C)).mat, C2)


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
    C_expected = SO3.rotz(y) * SO3.roty(p) * SO3.rotx(r)
    assert np.allclose(C_got.mat, C_expected.mat)


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
    assert np.allclose(C.as_matrix(), (SO3.exp(phi) * C_copy).as_matrix())


def test_normalize():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    C.mat += 0.1
    C.normalize()
    assert SO3.is_valid_matrix(C.mat)


def test_inv():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose((C * C.inv()).mat, np.identity(3))


def test_adjoint():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose(C.adjoint(), C.mat)


def test_transform_vectorized():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    pt1 = np.array([1, 2, 3])
    pt2 = np.array([4, 5, 3])
    pts = np.array([pt1, pt2]).T  # 3x2
    Cpt1 = C * pt1
    Cpt2 = C * pt2
    Cpts = C * pts
    assert(
        np.allclose(Cpt1, Cpts[:, 0])
        and np.allclose(Cpt2, Cpts[:, 1])
    )
