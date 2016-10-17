import numpy as np

from liegroups import SO3


def test_identity():
    T = SO3.identity()
    assert isinstance(T, SO3)


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
    phi = np.array([1, 2, 3])
    Phi = SO3.wedge(phi)
    assert np.array_equal(Phi, -Phi.T)


def test_wedge_vee():
    phi = np.array([1, 2, 3])
    Phi = SO3.wedge(phi)
    assert np.array_equal(phi, SO3.vee(Phi))


def test_exp_log():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    assert np.allclose(SO3.exp(SO3.log(C)).mat, C.mat)


def test_exp_log_zeros():
    C = SO3.exp(np.zeros(3))
    assert np.allclose(SO3.exp(SO3.log(C)).mat, C.mat)


def test_perturb():
    C = SO3.exp(np.pi * np.ones(3) / 4)
    C_copy = SO3.from_matrix(C.as_matrix())
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
