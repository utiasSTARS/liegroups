import copy

import torch
import numpy as np

from liegroups.torch import SO2, isclose, allclose


def test_identity():
    C = SO2.identity()
    assert isinstance(C, SO2) \
        and C.mat.dim() == 2 \
        and C.mat.shape == (2, 2)


def test_identity_batch():
    C = SO2.identity(5)
    assert isinstance(C, SO2) \
        and C.mat.dim() == 3 \
        and C.mat.shape == (5, 2, 2)


def test_from_angle_to_angle():
    angle = torch.Tensor([np.pi / 2.])
    assert allclose(SO2.from_angle(angle).to_angle(), angle)


def test_from_angle_to_angle_batch():
    angles = torch.Tensor([-1., 0, 1.])
    assert allclose(SO2.from_angle(angles).to_angle(), angles)


def test_dot():
    C = SO2(torch.Tensor([[0, -1],
                          [1, 0]]))
    pt = torch.Tensor([1, 2])

    CC = C.mat.mm(C.mat)
    assert allclose(C.dot(C).mat, CC)

    Cpt = C.mat.matmul(pt)
    assert allclose(C.dot(pt), Cpt)


def test_dot_batch():
    C1 = SO2(torch.Tensor([[0, -1],
                           [1, 0]]).expand(5, 2, 2))
    C2 = SO2(torch.Tensor([[-1, 0],
                           [0, -1]]))
    pt1 = torch.Tensor([1, 2])
    pt2 = torch.Tensor([4, 5])
    pts = torch.cat([pt1.unsqueeze(dim=1), pt2.unsqueeze(dim=1)], dim=1)  # 2x2

    C1C1 = torch.bmm(C1.mat, C1.mat)
    C1C1_SO2 = C1.dot(C1).mat
    assert C1C1_SO2.shape == C1.mat.shape and allclose(C1C1_SO2, C1C1)

    C1C2 = torch.matmul(C1.mat, C2.mat)
    C1C2_SO2 = C1.dot(C2).mat
    assert C1C2_SO2.shape == C1.mat.shape and allclose(C1C2_SO2, C1C2)

    C1pt1 = torch.matmul(C1.mat, pt1)
    C1pt1_SO2 = C1.dot(pt1)
    assert C1pt1_SO2.shape == (C1.mat.shape[0], pt1.shape[0]) \
        and allclose(C1pt1_SO2, C1pt1)

    C1pt2 = torch.matmul(C1.mat, pt2)
    C1pt2_SO2 = C1.dot(pt2)
    assert C1pt2_SO2.shape == (C1.mat.shape[0], pt2.shape[0]) \
        and allclose(C1pt2_SO2, C1pt2)

    C1pts = torch.matmul(C1.mat, pts)
    C1pts_SO2 = C1.dot(pts)
    assert C1pts_SO2.shape == (C1.mat.shape[0], pts.shape[0], pts.shape[1]) \
        and allclose(C1pts_SO2, C1pts) \
        and allclose(C1pt1, C1pts[:, :, 0]) \
        and allclose(C1pt2, C1pts[:, :, 1])


def test_wedge():
    phi = torch.Tensor([1])
    Phi = SO2.wedge(phi)
    assert (Phi == -Phi.t()).all()


def test_wedge_batch():
    phis = torch.Tensor([1, 2])
    Phis = SO2.wedge(phis)
    assert (Phis[0, :, :] == SO2.wedge(torch.Tensor([phis[0]]))).all()
    assert (Phis[1, :, :] == SO2.wedge(torch.Tensor([phis[1]]))).all()


def test_wedge_vee():
    phi = torch.Tensor([1])
    Phi = SO2.wedge(phi)
    assert (phi == SO2.vee(Phi)).all()


def test_wedge_vee_batch():
    phis = torch.Tensor([1, 2])
    Phis = SO2.wedge(phis)
    assert (phis == SO2.vee(Phis)).all()


def test_exp_log():
    C_big = SO2.exp(torch.Tensor([np.pi / 4]))
    assert allclose(SO2.exp(SO2.log(C_big)).mat, C_big.mat)

    C_small = SO2.exp(torch.Tensor([0]))
    assert allclose(SO2.exp(SO2.log(C_small)).mat, C_small.mat)


def test_exp_log_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_perturb():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C_copy = copy.deepcopy(C)
    phi = torch.Tensor([0.1])
    C.perturb(phi)
    assert allclose(C.as_matrix(), (SO2.exp(phi).dot(C_copy)).as_matrix())


def test_perturb_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    C_copy1 = copy.deepcopy(C)
    C_copy2 = copy.deepcopy(C)

    phi = torch.Tensor([0.1])
    C_copy1.perturb(phi)
    assert allclose(C_copy1.as_matrix(), (SO2.exp(phi).dot(C)).as_matrix())

    phis = torch.Tensor([0.1, 0.2, 0.3])
    C_copy2.perturb(phis)
    assert allclose(C_copy2.as_matrix(), (SO2.exp(phis).dot(C)).as_matrix())


def test_normalize():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C.mat.add_(0.1)
    C.normalize()
    assert SO2.is_valid_matrix(C.mat).all()


def test_normalize_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert (SO2.is_valid_matrix(C.mat) == torch.ByteTensor([1, 1, 1])).all()

    C.mat.add_(0.1)
    assert (SO2.is_valid_matrix(C.mat) == torch.ByteTensor([0, 0, 0])).all()

    C.normalize(inds=[0, 2])
    assert (SO2.is_valid_matrix(C.mat) == torch.ByteTensor([1, 0, 1])).all()

    C.normalize()
    assert SO2.is_valid_matrix(C.mat).all()


def test_inv():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert allclose(C.dot(C.inv()).mat, SO2.identity().mat)


def test_inv_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert allclose(C.dot(C.inv()).mat, SO2.identity(C.mat.shape[0]).mat)


def test_adjoint():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert (C.adjoint() == torch.Tensor([1.])).all()


def test_adjoint_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert (C.adjoint() == torch.ones(C.mat.shape[0])).all()
