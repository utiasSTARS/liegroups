import copy

import numpy as np

from liegroups import SO2

def test_bindto():
    C1 = SO2.identity()
    C2 = SO2.identity()
    C2.bindto(C1)
    assert(C1 is not C2 and C1.mat is C2.mat)


def test_identity():
    C = SO2.identity()
    assert isinstance(C, SO2)


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


def test_perturb():
    C = SO2.exp(np.pi / 4)
    C_copy = copy.deepcopy(C)
    phi = 0.1
    C.perturb(phi)
    assert np.allclose(C.as_matrix(), (SO2.exp(phi) * C_copy).as_matrix())


def test_normalize():
    C = SO2.exp(np.pi / 4)
    C.mat += 0.1
    C.normalize()
    assert SO2.is_valid_matrix(C.mat)


def test_inv():
    C = SO2.exp(np.pi / 4)
    assert np.allclose((C * C.inv()).mat, np.identity(2))


def test_adjoint():
    C = SO2.exp(np.pi / 4)
    assert C.adjoint() == 1.
