import numpy as np

from liegroups import SE2


def test_identity():
    T = SE2.identity()
    assert isinstance(T, SE2)


def test_mul():
    T = np.array([[0, -1, -0.5],
                  [1, 0, 0.5],
                  [0, 0, 1]])
    T2 = T.dot(T)
    assert np.allclose(
        (SE2.from_matrix(T) * SE2.from_matrix(T)).as_matrix(), T2)


def test_wedge_vee():
    xi = np.array([1, 2, 3])
    Xi = SE2.wedge(xi)
    assert np.array_equal(xi, SE2.vee(Xi))


def test_exp_log():
    T = SE2.exp(np.array([1, 2, 3]))
    assert np.allclose(SE2.exp(SE2.log(T)).as_matrix(), T.as_matrix())


def test_perturb():
    T = SE2.exp(np.array([1, 2, 3]))
    T_copy = SE2.from_matrix(T.as_matrix())
    xi = np.array([0.3, 0.2, 0.1])
    T.perturb(xi)
    assert np.allclose(T.as_matrix(), (SE2.exp(xi) * T_copy).as_matrix())


def test_normalize():
    T = SE2.exp(np.array([1, 2, 3]))
    T.rot.mat += 0.1
    T.normalize()
    assert SE2.is_valid_matrix(T.as_matrix())


def test_inv():
    T = SE2.exp(np.array([1, 2, 3]))
    assert np.allclose((T * T.inv()).as_matrix(), np.identity(3))


def test_adjoint():
    T = SE2.exp(np.array([1, 2, 3]))
    assert T.adjoint().shape == (3, 3)
