import copy

import torch
import numpy as np

from liegroups.torch import SO2, utils


def test_from_matrix():
    C_good = SO2.from_matrix(torch.eye(2))
    assert isinstance(C_good, SO2) \
        and C_good.mat.dim() == 2 \
        and C_good.mat.shape == (2, 2) \
        and SO2.is_valid_matrix(C_good.mat).all()

    C_bad = SO2.from_matrix(torch.eye(2).add_(1e-3), normalize=True)
    assert isinstance(C_bad, SO2) \
        and C_bad.mat.dim() == 2 \
        and C_bad.mat.shape == (2, 2) \
        and SO2.is_valid_matrix(C_bad.mat).all()


def test_from_matrix_batch():
    C_good = SO2.from_matrix(torch.eye(2).repeat(5, 1, 1))
    assert isinstance(C_good, SO2) \
        and C_good.mat.dim() == 3 \
        and C_good.mat.shape == (5, 2, 2) \
        and SO2.is_valid_matrix(C_good.mat).all()

    C_bad = copy.deepcopy(C_good.mat)
    C_bad[3].add_(0.1)
    C_bad = SO2.from_matrix(C_bad, normalize=True)
    assert isinstance(C_bad, SO2) \
        and C_bad.mat.dim() == 3 \
        and C_bad.mat.shape == (5, 2, 2) \
        and SO2.is_valid_matrix(C_bad.mat).all()


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

    C_copy = SO2.identity(5, copy=True)
    assert isinstance(C_copy, SO2) \
        and C_copy.mat.dim() == 3 \
        and C_copy.mat.shape == (5, 2, 2)


def test_from_angle_to_angle():
    angle = torch.Tensor([np.pi / 2.])
    assert utils.allclose(SO2.from_angle(angle).to_angle(), angle)


def test_from_angle_to_angle_batch():
    angles = torch.Tensor([-1., 0, 1.])
    assert utils.allclose(SO2.from_angle(angles).to_angle(), angles)


def test_dot():
    C = SO2(torch.Tensor([[0, -1],
                          [1, 0]]))
    pt = torch.Tensor([1, 2])

    CC = C.mat.mm(C.mat)
    assert utils.allclose(C.dot(C).mat, CC)

    Cpt = C.mat.matmul(pt)
    assert utils.allclose(C.dot(pt), Cpt)


def test_dot_batch():
    C1 = SO2(torch.Tensor([[0, -1],
                           [1, 0]]).expand(5, 2, 2))
    C2 = SO2(torch.Tensor([[-1, 0],
                           [0, -1]]))
    pt1 = torch.Tensor([1, 2])
    pt2 = torch.Tensor([4, 5])
    pt3 = torch.Tensor([7, 8])
    pts = torch.cat([pt1.unsqueeze(dim=0),
                     pt2.unsqueeze(dim=0),
                     pt3.unsqueeze(dim=0)], dim=0)  # 3x2
    ptsbatch = pts.unsqueeze(dim=0).expand(5, 3, 2)

    C1C1 = torch.bmm(C1.mat, C1.mat)
    C1C1_SO2 = C1.dot(C1).mat
    assert C1C1_SO2.shape == C1.mat.shape and utils.allclose(C1C1_SO2, C1C1)

    C1C2 = torch.matmul(C1.mat, C2.mat)
    C1C2_SO2 = C1.dot(C2).mat
    assert C1C2_SO2.shape == C1.mat.shape and utils.allclose(C1C2_SO2, C1C2)

    C1pt1 = torch.matmul(C1.mat, pt1)
    C1pt1_SO2 = C1.dot(pt1)
    assert C1pt1_SO2.shape == (C1.mat.shape[0], pt1.shape[0]) \
        and utils.allclose(C1pt1_SO2, C1pt1)

    C1pt2 = torch.matmul(C1.mat, pt2)
    C1pt2_SO2 = C1.dot(pt2)
    assert C1pt2_SO2.shape == (C1.mat.shape[0], pt2.shape[0]) \
        and utils.allclose(C1pt2_SO2, C1pt2)

    C1pts = torch.matmul(C1.mat, pts.transpose(1, 0)).transpose(2, 1)
    C1pts_SO2 = C1.dot(pts)
    assert C1pts_SO2.shape == (C1.mat.shape[0], pts.shape[0], pts.shape[1]) \
        and utils.allclose(C1pts_SO2, C1pts) \
        and utils.allclose(C1pt1, C1pts[:, 0, :]) \
        and utils.allclose(C1pt2, C1pts[:, 1, :])

    C1ptsbatch = torch.bmm(C1.mat, ptsbatch.transpose(2, 1)).transpose(2, 1)
    C1ptsbatch_SO2 = C1.dot(ptsbatch)
    assert C1ptsbatch_SO2.shape == ptsbatch.shape \
        and utils.allclose(C1ptsbatch_SO2, C1ptsbatch) \
        and utils.allclose(C1pt1, C1ptsbatch[:, 0, :]) \
        and utils.allclose(C1pt2, C1ptsbatch[:, 1, :])

    C2ptsbatch = torch.matmul(C2.mat, ptsbatch.transpose(2, 1)).transpose(2, 1)
    C2ptsbatch_SO2 = C2.dot(ptsbatch)
    assert C2ptsbatch_SO2.shape == ptsbatch.shape \
        and utils.allclose(C2ptsbatch_SO2, C2ptsbatch) \
        and utils.allclose(C2.dot(pt1), C2ptsbatch[:, 0, :]) \
        and utils.allclose(C2.dot(pt2), C2ptsbatch[:, 1, :])


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


def test_left_jacobians():
    phi_small = torch.Tensor([0.])
    phi_big = torch.Tensor([np.pi / 2])

    left_jacobian_small = SO2.left_jacobian(phi_small)
    inv_left_jacobian_small = SO2.inv_left_jacobian(phi_small)
    assert utils.allclose(
        torch.mm(left_jacobian_small, inv_left_jacobian_small),
        torch.eye(2))

    left_jacobian_big = SO2.left_jacobian(phi_big)
    inv_left_jacobian_big = SO2.inv_left_jacobian(phi_big)
    assert utils.allclose(
        torch.mm(left_jacobian_big, inv_left_jacobian_big),
        torch.eye(2))


def test_left_jacobians_batch():
    phis = torch.Tensor([0., np.pi / 2])

    left_jacobian = SO2.left_jacobian(phis)
    inv_left_jacobian = SO2.inv_left_jacobian(phis)
    assert utils.allclose(torch.bmm(left_jacobian, inv_left_jacobian),
                          torch.eye(2).unsqueeze_(dim=0).expand(2, 2, 2))


def test_exp_log():
    C_big = SO2.exp(torch.Tensor([np.pi / 4]))
    assert utils.allclose(SO2.exp(SO2.log(C_big)).mat, C_big.mat)

    C_small = SO2.exp(torch.Tensor([0]))
    assert utils.allclose(SO2.exp(SO2.log(C_small)).mat, C_small.mat)


def test_exp_log_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert utils.allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_perturb():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C_copy = copy.deepcopy(C)
    phi = torch.Tensor([0.1])
    C.perturb(phi)
    assert utils.allclose(
        C.as_matrix(), (SO2.exp(phi).dot(C_copy)).as_matrix())


def test_perturb_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    C_copy1 = copy.deepcopy(C)
    C_copy2 = copy.deepcopy(C)

    phi = torch.Tensor([0.1])
    C_copy1.perturb(phi)
    assert utils.allclose(C_copy1.as_matrix(),
                          (SO2.exp(phi).dot(C)).as_matrix())

    phis = torch.Tensor([0.1, 0.2, 0.3])
    C_copy2.perturb(phis)
    assert utils.allclose(C_copy2.as_matrix(),
                          (SO2.exp(phis).dot(C)).as_matrix())


def test_normalize():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C.mat.add_(0.1)
    C.normalize()
    assert SO2.is_valid_matrix(C.mat).all()


def test_normalize_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert SO2.is_valid_matrix(C.mat).all()

    C.mat.add_(0.1)
    assert (SO2.is_valid_matrix(C.mat) == torch.ByteTensor([0, 0, 0])).all()

    C.normalize(inds=[0, 2])
    assert (SO2.is_valid_matrix(C.mat) == torch.ByteTensor([1, 0, 1])).all()

    C.normalize()
    assert SO2.is_valid_matrix(C.mat).all()


def test_inv():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert utils.allclose(C.dot(C.inv()).mat, SO2.identity().mat)


def test_inv_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert utils.allclose(C.dot(C.inv()).mat, SO2.identity(C.mat.shape[0]).mat)


def test_adjoint():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert (C.adjoint() == torch.Tensor([1.])).all()


def test_adjoint_batch():
    C = SO2.exp(torch.Tensor([-1., 0., 1.]))
    assert (C.adjoint() == torch.ones(C.mat.shape[0])).all()
