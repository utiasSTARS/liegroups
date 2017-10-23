import copy

import numpy as np

from liegroups.numpy import SE3


def test_identity():
    T = SE3.identity()
    assert isinstance(T, SE3)


def test_dot():
    T = np.array([[0, 0, -1, 0.1],
                  [0, 1, 0, 0.5],
                  [1, 0, 0, -0.5],
                  [0, 0, 0, 1]])
    T2 = T.dot(T)
    assert np.allclose(
        (SE3.from_matrix(T).dot(SE3.from_matrix(T))).as_matrix(), T2)


def test_wedge_vee():
    xi = [1, 2, 3, 4, 5, 6]
    Xi = SE3.wedge(xi)
    xis = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    Xis = SE3.wedge(xis)
    assert np.array_equal(xi, SE3.vee(Xi))
    assert np.array_equal(xis, SE3.vee(Xis))


def test_curlywedge_curlyvee():
    xi = [1, 2, 3, 4, 5, 6]
    Psi = SE3.curlywedge(xi)
    xis = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    Psis = SE3.curlywedge(xis)
    assert np.array_equal(xi, SE3.curlyvee(Psi))
    assert np.array_equal(xis, SE3.curlyvee(Psis))


def test_odot():
    p1 = [1, 2, 3]
    p2 = [1, 2, 3, 1]
    p3 = [1, 2, 3, 0]

    odot12 = np.vstack([SE3.odot(p1), np.zeros([1, 6])])
    odot13 = np.vstack([SE3.odot(p1, directional=True), np.zeros([1, 6])])
    odot2 = SE3.odot(p2)
    odot3 = SE3.odot(p3)

    assert np.array_equal(odot12, odot2)
    assert np.array_equal(odot13, odot3)


def test_odot_vectorized():
    p1 = [1, 2, 3]
    p2 = [2, 3, 4]
    ps = np.array([p1, p2])

    odot1 = SE3.odot(p1)
    odot2 = SE3.odot(p2)
    odots = SE3.odot(ps)

    assert np.array_equal(odot1, odots[0, :, :])
    assert np.array_equal(odot2, odots[1, :, :])


def test_exp_log():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    assert np.allclose(SE3.exp(SE3.log(T)).as_matrix(), T.as_matrix())


def test_left_jacobian():
    xi1 = [1, 2, 3, 4, 5, 6]
    assert np.allclose(
        SE3.left_jacobian(xi1).dot(SE3.inv_left_jacobian(xi1)),
        np.identity(6)
    )

    xi2 = [0, 0, 0, 0, 0, 0]
    assert np.allclose(
        SE3.left_jacobian(xi2).dot(SE3.inv_left_jacobian(xi2)),
        np.identity(6)
    )


def test_perturb():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    T_copy = copy.deepcopy(T)
    xi = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    T.perturb(xi)
    assert np.allclose(T.as_matrix(), (SE3.exp(xi).dot(T_copy)).as_matrix())


def test_normalize():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    T.rot.mat += 0.1
    T.normalize()
    assert SE3.is_valid_matrix(T.as_matrix())


def test_inv():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    assert np.allclose((T.dot(T.inv())).as_matrix(), np.identity(4))


def test_adjoint():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    assert T.adjoint().shape == (6, 6)


def test_transform_vectorized():
    T = SE3.exp([1, 2, 3, 4, 5, 6])
    pt1 = np.array([1, 2, 3])
    pt2 = np.array([4, 5, 6])
    pt3 = np.array([1, 2, 3, 1])
    pt4 = np.array([4, 5, 6, 1])
    pts12 = np.array([pt1, pt2])  # 2x3
    pts34 = np.array([pt3, pt4])  # 2x4
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
