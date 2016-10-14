import numpy as np

from liegroups import SO2


def test_mul():
    C = np.array([[0, -1],
                  [1, 0]])
    C2 = C.dot(C)
    assert np.allclose((SO2(C) * SO2(C)).mat, C2)


def test_wedge():
    phi = 1
    Phi = SO2.wedge(phi)
    assert np.array_equal(Phi, -Phi.T)


def test_wedge_vee():
    phi = 1
    Phi = SO2.wedge(phi)
    assert phi == SO2.vee(Phi)


def test_exp_log():
    C = SO2.exp(np.pi / 4)
    assert np.allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_exp_log_zeros():
    C = SO2.exp(0)
    assert np.allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_normalize():
    C = SO2.exp(np.pi / 4)
    C.mat += 0.1
    C.normalize()
    assert SO2.is_valid_matrix(C.mat)


def test_inverse():
    C = SO2.exp(np.pi / 4)
    assert np.allclose((C * C.inverse()).mat, np.identity(2))


def test_adjoint():
    C = SO2.exp(np.pi / 4)
    assert C.adjoint() == 1.
