import numpy as np

from liegroups import SE3


def test_mul():
    T = np.array([[0, 0, -1, 0.1],
                  [0, 1, 0, 0.5],
                  [1, 0, 0, -0.5],
                  [0, 0, 0, 1]])
    T2 = T.dot(T)
    assert np.allclose((SE3.frommatrix(T) * SE3.frommatrix(T)).asmatrix(), T2)


def test_wedge_vee():
    xi = np.array([1, 2, 3, 4, 5, 6])
    Xi = SE3.wedge(xi)
    assert np.array_equal(xi, SE3.vee(Xi))


def test_exp_log():
    T = SE3.exp(np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose(SE3.exp(SE3.log(T)).asmatrix(), T.asmatrix())


def test_normalize():
    T = SE3.exp(np.array([1, 2, 3, 4, 5, 6]))
    T.rot.mat += 0.1
    T.normalize()
    assert SE3.isvalidmatrix(T.asmatrix())


def test_inverse():
    T = SE3.exp(np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose((T * T.inverse()).asmatrix(), np.identity(4))


def test_adjoint():
    T = SE3.exp(np.array([1, 2, 3, 4, 5, 6]))
    assert T.adjoint().shape == (6, 6)
