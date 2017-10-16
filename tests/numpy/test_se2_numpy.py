import copy

import numpy as np

from liegroups.numpy import SE2


def test_identity():
    T = SE2.identity()
    assert isinstance(T, SE2)


def test_dot():
    T = np.array([[0, -1, -0.5],
                  [1, 0, 0.5],
                  [0, 0, 1]])
    T2 = T.dot(T)
    assert np.allclose(
        (SE2.from_matrix(T).dot(SE2.from_matrix(T))).as_matrix(), T2)


def test_wedge_vee():
    xi = [1, 2, 3]
    Xi = SE2.wedge(xi)
    xis = np.array([[1, 2, 3], [4, 5, 6]])
    Xis = SE2.wedge(xis)
    assert np.array_equal(xi, SE2.vee(Xi))
    assert np.array_equal(xis, SE2.vee(Xis))


def test_odot():
    p1 = [1, 2]
    p2 = [1, 2, 1]
    p3 = [1, 2, 0]

    odot12 = np.vstack([SE2.odot(p1), np.zeros([1, 3])])
    odot13 = np.vstack([SE2.odot(p1, directional=True), np.zeros([1, 3])])
    odot2 = SE2.odot(p2)
    odot3 = SE2.odot(p3)

    assert np.array_equal(odot12, odot2)
    assert np.array_equal(odot13, odot3)


def test_odot_vectorized():
    p1 = [1, 2]
    p2 = [2, 3]
    ps = np.array([p1, p2])

    odot1 = SE2.odot(p1)
    odot2 = SE2.odot(p2)
    odots = SE2.odot(ps)

    assert np.array_equal(odot1, odots[0, :, :])
    assert np.array_equal(odot2, odots[1, :, :])


def test_exp_log():
    T = SE2.exp([1, 2, 3])
    assert np.allclose(SE2.exp(SE2.log(T)).as_matrix(), T.as_matrix())


def test_perturb():
    T = SE2.exp([1, 2, 3])
    T_copy = copy.deepcopy(T)
    xi = [0.3, 0.2, 0.1]
    T.perturb(xi)
    assert np.allclose(T.as_matrix(), (SE2.exp(xi).dot(T_copy)).as_matrix())


def test_normalize():
    T = SE2.exp([1, 2, 3])
    T.rot.mat += 0.1
    T.normalize()
    assert SE2.is_valid_matrix(T.as_matrix())


def test_inv():
    T = SE2.exp([1, 2, 3])
    assert np.allclose((T.dot(T.inv())).as_matrix(), np.identity(3))


def test_adjoint():
    T = SE2.exp([1, 2, 3])
    assert T.adjoint().shape == (3, 3)


def test_transform_vectorized():
    T = SE2.exp([1, 2, 3])
    pt1 = np.array([1, 2])
    pt2 = np.array([4, 5])
    pt3 = np.array([1, 2, 1])
    pt4 = np.array([4, 5, 1])
    pts12 = np.array([pt1, pt2])  # 2x2
    pts34 = np.array([pt3, pt4])  # 2x3
    Tpt1 = T.dot(pt1)
    Tpt2 = T.dot(pt2)
    Tpt3 = T.dot(pt3)
    Tpt4 = T.dot(pt4)
    Tpts12 = T.dot(pts12)
    Tpts34 = T.dot(pts34)
    assert np.allclose(Tpt1, Tpts12[0])
    assert np.allclose(Tpt2, Tpts12[1])
    assert np.allclose(Tpt3, Tpts34[0])
    assert np.allclose(Tpt4, Tpts34[1])
