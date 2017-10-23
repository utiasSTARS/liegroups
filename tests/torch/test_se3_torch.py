import copy

import torch

from liegroups.torch import SE3, SO3, utils


def test_from_matrix():
    T_good = SE3.from_matrix(torch.eye(4))
    assert isinstance(T_good, SE3) \
        and isinstance(T_good.rot, SO3) \
        and T_good.trans.shape == (3,) \
        and SE3.is_valid_matrix(T_good.as_matrix()).all()

    T_bad = SE3.from_matrix(torch.eye(4).add_(1e-3), normalize=True)
    assert isinstance(T_bad, SE3) \
        and isinstance(T_bad.rot, SO3) \
        and T_bad.trans.shape == (3,) \
        and SE3.is_valid_matrix(T_bad.as_matrix()).all()


def test_from_matrix_batch():
    T_good = SE3.from_matrix(torch.eye(4).repeat(5, 1, 1))
    assert isinstance(T_good, SE3) \
        and T_good.trans.shape == (5, 3) \
        and SE3.is_valid_matrix(T_good.as_matrix()).all()

    T_bad = T_good.as_matrix()
    T_bad[3, :, :].add_(0.1)
    T_bad = SE3.from_matrix(T_bad, normalize=True)
    assert isinstance(T_bad, SE3) \
        and T_bad.trans.shape == (5, 3) \
        and SE3.is_valid_matrix(T_bad.as_matrix()).all()


def test_identity():
    T = SE3.identity()
    assert isinstance(T, SE3) \
        and isinstance(T.rot, SO3) \
        and T.rot.mat.dim() == 2 \
        and T.trans.shape == (3,)


def test_identity_batch():
    T = SE3.identity(5)
    assert isinstance(T, SE3) \
        and isinstance(T.rot, SO3) \
        and T.rot.mat.dim() == 3 \
        and T.trans.shape == (5, 3)


def test_dot():
    T = torch.Tensor([[0, 0, -1, 0.1],
                      [0, 1, 0, 0.5],
                      [1, 0, 0, -0.5],
                      [0, 0, 0, 1]])
    T_SE3 = SE3.from_matrix(T)
    pt = torch.Tensor([1, 2, 3])
    pth = torch.Tensor([1, 2, 3, 1])

    TT = torch.mm(T, T)
    TT_SE3 = T_SE3.dot(T_SE3).as_matrix()
    assert utils.allclose(TT_SE3, TT)

    Tpt = torch.matmul(T[0:3, 0:3], pt) + T[0:3, 3]
    Tpt_SE3 = T_SE3.dot(pt)
    assert utils.allclose(Tpt_SE3, Tpt)

    Tpth = torch.matmul(T, pth)
    Tpth_SE3 = T_SE3.dot(pth)
    assert utils.allclose(Tpth_SE3, Tpth) and \
        utils.allclose(Tpth_SE3[0:3], Tpt)


def test_dot_batch():
    T1 = torch.Tensor([[0, 0, -1, 0.1],
                       [0, 1, 0, 0.5],
                       [1, 0, 0, -0.5],
                       [0, 0, 0, 1]]).expand(5, 4, 4)
    T2 = torch.Tensor([[0, 0, -1, 0.1],
                       [0, 1, 0, 0.5],
                       [1, 0, 0, -0.5],
                       [0, 0, 0, 1]])
    T1_SE3 = SE3.from_matrix(T1)
    T2_SE3 = SE3.from_matrix(T2)
    pt1 = torch.Tensor([1, 2, 3])
    pt2 = torch.Tensor([4, 5, 6])
    pt3 = torch.Tensor([7, 8, 9])
    pts = torch.cat([pt1.unsqueeze(dim=0),
                     pt2.unsqueeze(dim=0),
                     pt3.unsqueeze(dim=0)], dim=0)  # 3x3
    ptsbatch = pts.unsqueeze(dim=0).expand(5, 3, 3)
    pt1h = torch.Tensor([1, 2, 3, 1])
    pt2h = torch.Tensor([4, 5, 6, 1])
    pt3h = torch.Tensor([7, 8, 9, 1])
    ptsh = torch.cat([pt1h.unsqueeze(dim=0),
                      pt2h.unsqueeze(dim=0),
                      pt3h.unsqueeze(dim=0)], dim=0)  # 3x4
    ptshbatch = ptsh.unsqueeze(dim=0).expand(5, 3, 4)

    T1T1 = torch.bmm(T1, T1)
    T1T1_SE3 = T1_SE3.dot(T1_SE3).as_matrix()
    assert T1T1_SE3.shape == T1.shape and utils.allclose(T1T1_SE3, T1T1)

    T1T2 = torch.matmul(T1, T2)
    T1T2_SE3 = T1_SE3.dot(T2_SE3).as_matrix()
    assert T1T2_SE3.shape == T1.shape and utils.allclose(T1T2_SE3, T1T2)

    T1pt1 = torch.matmul(T1[:, 0:3, 0:3], pt1) + T1[:, 0:3, 3]
    T1pt1_SE3 = T1_SE3.dot(pt1)
    assert T1pt1_SE3.shape == (T1.shape[0], pt1.shape[0]) \
        and utils.allclose(T1pt1_SE3, T1pt1)

    T1pt1h = torch.matmul(T1, pt1h)
    T1pt1h_SE3 = T1_SE3.dot(pt1h)
    assert T1pt1h_SE3.shape == (T1.shape[0], pt1h.shape[0]) \
        and utils.allclose(T1pt1h_SE3, T1pt1h) \
        and utils.allclose(T1pt1h_SE3[:, 0:3], T1pt1_SE3)

    T1pt2 = torch.matmul(T1[:, 0:3, 0:3], pt2) + T1[:, 0:3, 3]
    T1pt2_SE3 = T1_SE3.dot(pt2)
    assert T1pt2_SE3.shape == (T1.shape[0], pt2.shape[0]) \
        and utils.allclose(T1pt2_SE3, T1pt2)

    T1pt2h = torch.matmul(T1, pt2h)
    T1pt2h_SE3 = T1_SE3.dot(pt2h)
    assert T1pt2h_SE3.shape == (T1.shape[0], pt2h.shape[0]) \
        and utils.allclose(T1pt2h_SE3, T1pt2h) \
        and utils.allclose(T1pt2h_SE3[:, 0:3], T1pt2_SE3)

    T1pts = torch.bmm(T1[:, 0:3, 0:3],
                      pts.unsqueeze(dim=0).expand(
                          T1.shape[0],
                          pts.shape[0],
                          pts.shape[1]).transpose(2, 1)).transpose(2, 1) + \
        T1[:, 0:3, 3].unsqueeze(dim=1).expand(
            T1.shape[0], pts.shape[0], pts.shape[1])
    T1pts_SE3 = T1_SE3.dot(pts)
    assert T1pts_SE3.shape == (T1.shape[0], pts.shape[0], pts.shape[1]) \
        and utils.allclose(T1pts_SE3, T1pts) \
        and utils.allclose(T1pt1, T1pts[:, 0, :]) \
        and utils.allclose(T1pt2, T1pts[:, 1, :])

    T1ptsh = torch.bmm(T1, ptsh.unsqueeze(dim=0).expand(
        T1.shape[0],
        ptsh.shape[0],
        ptsh.shape[1]).transpose(2, 1)).transpose(2, 1)
    T1ptsh_SE3 = T1_SE3.dot(ptsh)
    assert T1ptsh_SE3.shape == (T1.shape[0], ptsh.shape[0], ptsh.shape[1]) \
        and utils.allclose(T1ptsh_SE3, T1ptsh) \
        and utils.allclose(T1pt1h, T1ptsh[:, 0, :]) \
        and utils.allclose(T1pt2h, T1ptsh[:, 1, :]) \
        and utils.allclose(T1ptsh_SE3[:, :, 0:3], T1pts_SE3)

    T1ptsbatch = torch.bmm(T1[:, 0:3, 0:3],
                           ptsbatch.transpose(2, 1)).transpose(2, 1) + \
        T1[:, 0:3, 3].unsqueeze(dim=1).expand(ptsbatch.shape)
    T1ptsbatch_SE3 = T1_SE3.dot(ptsbatch)
    assert T1ptsbatch_SE3.shape == ptsbatch.shape \
        and utils.allclose(T1ptsbatch_SE3, T1ptsbatch) \
        and utils.allclose(T1pt1, T1ptsbatch[:, 0, :]) \
        and utils.allclose(T1pt2, T1ptsbatch[:, 1, :])

    T1ptshbatch = torch.bmm(T1, ptshbatch.transpose(2, 1)).transpose(2, 1)
    T1ptshbatch_SE3 = T1_SE3.dot(ptshbatch)
    assert T1ptshbatch_SE3.shape == ptshbatch.shape \
        and utils.allclose(T1ptshbatch_SE3, T1ptshbatch) \
        and utils.allclose(T1pt1h, T1ptshbatch[:, 0, :]) \
        and utils.allclose(T1pt2h, T1ptshbatch[:, 1, :]) \
        and utils.allclose(T1ptshbatch_SE3[:, :, 0:3], T1ptsbatch_SE3)

    T2ptsbatch = torch.matmul(T2[0:3, 0:3],
                              ptsbatch.transpose(2, 1)).transpose(2, 1) + \
        T1[:, 0:3, 3].unsqueeze(dim=1).expand(ptsbatch.shape)
    T2ptsbatch_SE3 = T2_SE3.dot(ptsbatch)
    assert T2ptsbatch_SE3.shape == ptsbatch.shape \
        and utils.allclose(T2ptsbatch_SE3, T2ptsbatch) \
        and utils.allclose(T2_SE3.dot(pt1), T2ptsbatch[:, 0, :]) \
        and utils.allclose(T2_SE3.dot(pt2), T2ptsbatch[:, 1, :])

    T2ptshbatch = torch.matmul(T2, ptshbatch.transpose(2, 1)).transpose(2, 1)
    T2ptshbatch_SE3 = T2_SE3.dot(ptshbatch)
    assert T2ptshbatch_SE3.shape == ptshbatch.shape \
        and utils.allclose(T2ptshbatch_SE3, T2ptshbatch) \
        and utils.allclose(T2_SE3.dot(pt1h), T2ptshbatch[:, 0, :]) \
        and utils.allclose(T2_SE3.dot(pt2h), T2ptshbatch[:, 1, :]) \
        and utils.allclose(T2ptshbatch_SE3[:, :, 0:3], T2ptsbatch_SE3)


def test_wedge_vee():
    xi = 0.1 * torch.Tensor([1, 2, 3, 4, 5, 6])
    Xi = SE3.wedge(xi)
    assert (xi == SE3.vee(Xi)).all()


def test_wedge_vee_batch():
    xis = 0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    Xis = SE3.wedge(xis)
    assert (xis == SE3.vee(Xis)).all()


def test_curlywedge_curlyvee():
    xi = torch.Tensor([1, 2, 3, 4, 5, 6])
    Psi = SE3.curlywedge(xi)
    assert (xi == SE3.curlyvee(Psi)).all()


def test_curlywedge_curlyvee_batch():
    xis = torch.Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    Psis = SE3.curlywedge(xis)
    assert (xis == SE3.curlyvee(Psis)).all()


def test_odot():
    p1 = torch.Tensor([1, 2, 3])
    p2 = torch.Tensor([1, 2, 3, 1])
    p3 = torch.Tensor([1, 2, 3, 0])

    odot12 = torch.cat([SE3.odot(p1), torch.zeros(6).unsqueeze_(dim=0)], dim=0)
    odot13 = torch.cat([SE3.odot(p1, directional=True),
                        torch.zeros(6).unsqueeze_(dim=0)], dim=0)
    odot2 = SE3.odot(p2)
    odot3 = SE3.odot(p3)

    assert (odot12 == odot2).all()
    assert (odot13 == odot3).all()


def test_odot_batch():
    p1 = torch.Tensor([1, 2, 3])
    p2 = torch.Tensor([4, 5, 6])
    ps = torch.cat([p1.unsqueeze(dim=0),
                    p2.unsqueeze(dim=0)], dim=0)

    odot1 = SE3.odot(p1)
    odot2 = SE3.odot(p2)
    odots = SE3.odot(ps)

    assert (odot1 == odots[0, :, :]).all()
    assert (odot2 == odots[1, :, :]).all()


def test_exp_log():
    T = SE3.exp(torch.Tensor([1, 2, 3, 4, 5, 6]))
    assert utils.allclose(SE3.exp(SE3.log(T)).as_matrix(), T.as_matrix())


def test_exp_log_batch():
    T = SE3.exp(0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12]]))
    assert utils.allclose(SE3.exp(SE3.log(T)).as_matrix(), T.as_matrix())


def test_left_jacobian():
    xi1 = torch.Tensor([1, 2, 3, 4, 5, 6])
    assert utils.allclose(
        torch.mm(SE3.left_jacobian(xi1), SE3.inv_left_jacobian(xi1)),
        torch.eye(6)
    )

    xi2 = torch.Tensor([0, 0, 0, 0, 0, 0])
    assert utils.allclose(
        torch.mm(SE3.left_jacobian(xi2), SE3.inv_left_jacobian(xi2)),
        torch.eye(6)
    )


def test_left_jacobian_batch():
    xis = torch.Tensor([[1, 2, 3, 4, 5, 6],
                        [0, 0, 0, 0, 0, 0]])
    assert utils.allclose(
        SE3.left_jacobian(xis).bmm(SE3.inv_left_jacobian(xis)),
        torch.eye(6).unsqueeze_(dim=0).expand(2, 6, 6)
    )


def test_perturb():
    T = SE3.exp(torch.Tensor([1, 2, 3, 4, 5, 6]))
    T_copy = copy.deepcopy(T)
    xi = torch.Tensor([0.3, 0.2, 0.1, -0.1, -0.2, -0.3])
    T.perturb(xi)
    assert utils.allclose(T.as_matrix(), (SE3.exp(xi).dot(T_copy)).as_matrix())


def test_perturb_batch():
    T = SE3.exp(0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12]]))
    T_copy1 = copy.deepcopy(T)
    T_copy2 = copy.deepcopy(T)

    xi = torch.Tensor([0.3, 0.2, 0.1, -0.1, -0.2, -0.3])
    T_copy1.perturb(xi)
    assert utils.allclose(T_copy1.as_matrix(),
                          (SE3.exp(xi).dot(T)).as_matrix())

    xis = torch.Tensor([[0.3, 0.2, 0.1, -0.1, -0.2, -0.3],
                        [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]])
    T_copy2.perturb(xis)
    assert utils.allclose(T_copy2.as_matrix(),
                          (SE3.exp(xis).dot(T)).as_matrix())


def test_normalize():
    T = SE3.exp(torch.Tensor([1, 2, 3, 4, 5, 6]))
    T.rot.mat.add_(0.1)
    T.normalize()
    assert SE3.is_valid_matrix(T.as_matrix()).all()


def test_normalize_batch():
    T = SE3.exp(0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12],
                                    [13, 14, 15, 16, 17, 18]]))
    assert SE3.is_valid_matrix(T.as_matrix()).all()

    T.rot.mat.add_(0.1)
    assert (SE3.is_valid_matrix(T.as_matrix())
            == torch.ByteTensor([0, 0, 0])).all()

    T.normalize(inds=[0, 2])
    assert (SE3.is_valid_matrix(T.as_matrix())
            == torch.ByteTensor([1, 0, 1])).all()

    T.normalize()
    assert SE3.is_valid_matrix(T.as_matrix()).all()


def test_inv():
    T = SE3.exp(torch.Tensor([1, 2, 3, 4, 5, 6]))
    assert utils.allclose((T.dot(T.inv())).as_matrix(), torch.eye(4))


def test_inv_batch():
    T = SE3.exp(0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12],
                                    [13, 14, 15, 16, 17, 18]]))
    assert utils.allclose(T.dot(T.inv()).as_matrix(),
                          SE3.identity(T.trans.shape[0]).as_matrix())


def test_adjoint():
    T = SE3.exp(torch.Tensor([1, 2, 3, 4, 5, 6]))
    assert T.adjoint().shape == (6, 6)


def test_adjoint_batch():
    T = SE3.exp(0.1 * torch.Tensor([[1, 2, 3, 4, 5, 6],
                                    [7, 8, 9, 10, 11, 12]]))
    assert T.adjoint().shape == (2, 6, 6)
