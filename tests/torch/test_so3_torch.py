import copy

import torch
import numpy as np

from liegroups.torch import SO3, utils


def test_from_matrix():
    C_good = SO3.from_matrix(torch.eye(3))
    assert isinstance(C_good, SO3) \
        and C_good.mat.dim() == 2 \
        and C_good.mat.shape == (3, 3) \
        and SO3.is_valid_matrix(C_good.mat).all()

    C_bad = SO3.from_matrix(torch.eye(3).add_(1e-3), normalize=True)
    assert isinstance(C_bad, SO3) \
        and C_bad.mat.dim() == 2 \
        and C_bad.mat.shape == (3, 3) \
        and SO3.is_valid_matrix(C_bad.mat).all()


def test_from_matrix_batch():
    C_good = SO3.from_matrix(torch.eye(3).repeat(5, 1, 1))
    assert isinstance(C_good, SO3) \
        and C_good.mat.dim() == 3 \
        and C_good.mat.shape == (5, 3, 3) \
        and SO3.is_valid_matrix(C_good.mat).all()

    C_bad = copy.deepcopy(C_good.mat)
    C_bad[3].add_(0.1)
    C_bad = SO3.from_matrix(C_bad, normalize=True)
    assert isinstance(C_bad, SO3) \
        and C_bad.mat.dim() == 3 \
        and C_bad.mat.shape == (5, 3, 3) \
        and SO3.is_valid_matrix(C_bad.mat).all()


def test_identity():
    C = SO3.identity()
    assert isinstance(C, SO3) \
        and C.mat.dim() == 2 \
        and C.mat.shape == (3, 3)


def test_identity_batch():
    C = SO3.identity(5)
    assert isinstance(C, SO3) \
        and C.mat.dim() == 3 \
        and C.mat.shape == (5, 3, 3)

    C_copy = SO3.identity(5, copy=True)
    assert isinstance(C_copy, SO3) \
        and C_copy.mat.dim() == 3 \
        and C_copy.mat.shape == (5, 3, 3)


def test_dot():
    C = SO3(torch.Tensor([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]]))
    pt = torch.Tensor([1, 2, 3])

    CC = C.mat.mm(C.mat)
    assert utils.allclose(C.dot(C).mat, CC)

    Cpt = C.mat.matmul(pt)
    assert utils.allclose(C.dot(pt), Cpt)


def test_dot_batch():
    C1 = SO3(torch.Tensor([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]]).expand(5, 3, 3))
    C3 = SO3(torch.Tensor([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]]))
    pt1 = torch.Tensor([1, 2, 3])
    pt3 = torch.Tensor([4, 5, 6])
    pt3 = torch.Tensor([7, 8, 9])
    pts = torch.cat([pt1.unsqueeze(dim=0),
                     pt3.unsqueeze(dim=0),
                     pt3.unsqueeze(dim=0)], dim=0)  # 3x3
    ptsbatch = pts.unsqueeze(dim=0).expand(5, 3, 3)

    C1C1 = torch.bmm(C1.mat, C1.mat)
    C1C1_SO3 = C1.dot(C1).mat
    assert C1C1_SO3.shape == C1.mat.shape and utils.allclose(C1C1_SO3, C1C1)

    C1C3 = torch.matmul(C1.mat, C3.mat)
    C1C3_SO3 = C1.dot(C3).mat
    assert C1C3_SO3.shape == C1.mat.shape and utils.allclose(C1C3_SO3, C1C3)

    C1pt1 = torch.matmul(C1.mat, pt1)
    C1pt1_SO3 = C1.dot(pt1)
    assert C1pt1_SO3.shape == (C1.mat.shape[0], pt1.shape[0]) \
        and utils.allclose(C1pt1_SO3, C1pt1)

    C1pt3 = torch.matmul(C1.mat, pt3)
    C1pt3_SO3 = C1.dot(pt3)
    assert C1pt3_SO3.shape == (C1.mat.shape[0], pt3.shape[0]) \
        and utils.allclose(C1pt3_SO3, C1pt3)

    C1pts = torch.matmul(C1.mat, pts.transpose(1, 0)).transpose(2, 1)
    C1pts_SO3 = C1.dot(pts)
    assert C1pts_SO3.shape == (C1.mat.shape[0], pts.shape[0], pts.shape[1]) \
        and utils.allclose(C1pts_SO3, C1pts) \
        and utils.allclose(C1pt1, C1pts[:, 0, :]) \
        and utils.allclose(C1pt3, C1pts[:, 1, :])

    C1ptsbatch = torch.bmm(C1.mat, ptsbatch.transpose(2, 1)).transpose(2, 1)
    C1ptsbatch_SO3 = C1.dot(ptsbatch)
    assert C1ptsbatch_SO3.shape == ptsbatch.shape \
        and utils.allclose(C1ptsbatch_SO3, C1ptsbatch) \
        and utils.allclose(C1pt1, C1ptsbatch[:, 0, :]) \
        and utils.allclose(C1pt3, C1ptsbatch[:, 1, :])

    C3ptsbatch = torch.matmul(C3.mat, ptsbatch.transpose(2, 1)).transpose(2, 1)
    C3ptsbatch_SO3 = C3.dot(ptsbatch)
    assert C3ptsbatch_SO3.shape == ptsbatch.shape \
        and utils.allclose(C3ptsbatch_SO3, C3ptsbatch) \
        and utils.allclose(C3.dot(pt1), C3ptsbatch[:, 0, :]) \
        and utils.allclose(C3.dot(pt3), C3ptsbatch[:, 1, :])


def test_wedge():
    phi = torch.Tensor([1, 2, 3])
    Phi = SO3.wedge(phi)
    assert (Phi == -Phi.t()).all()


def test_wedge_batch():
    phis = torch.Tensor([[1, 2, 3],
                         [4, 5, 6]])
    Phis = SO3.wedge(phis)
    assert (Phis[0, :, :] == SO3.wedge(phis[0])).all()
    assert (Phis[1, :, :] == SO3.wedge(phis[1])).all()


def test_wedge_vee():
    phi = torch.Tensor([1, 2, 3])
    Phi = SO3.wedge(phi)
    assert (phi == SO3.vee(Phi)).all()


def test_wedge_vee_batch():
    phis = torch.Tensor([[1, 2, 3],
                         [4, 5, 6]])
    Phis = SO3.wedge(phis)
    assert (phis == SO3.vee(Phis)).all()


def test_left_jacobians():
    phi_small = torch.Tensor([0., 0., 0.])
    phi_big = torch.Tensor([np.pi / 2, np.pi / 3, np.pi / 4])

    left_jacobian_small = SO3.left_jacobian(phi_small)
    inv_left_jacobian_small = SO3.inv_left_jacobian(phi_small)
    assert utils.allclose(
        torch.mm(left_jacobian_small, inv_left_jacobian_small),
        torch.eye(3))

    left_jacobian_big = SO3.left_jacobian(phi_big)
    inv_left_jacobian_big = SO3.inv_left_jacobian(phi_big)
    assert utils.allclose(
        torch.mm(left_jacobian_big, inv_left_jacobian_big),
        torch.eye(3))


def test_left_jacobians_batch():
    phis = torch.Tensor([[0., 0., 0.],
                         [np.pi / 2, np.pi / 3, np.pi / 4]])

    left_jacobian = SO3.left_jacobian(phis)
    inv_left_jacobian = SO3.inv_left_jacobian(phis)
    assert utils.allclose(torch.bmm(left_jacobian, inv_left_jacobian),
                          torch.eye(3).unsqueeze_(dim=0).expand(2, 3, 3))


def test_exp_log():
    C_big = SO3.exp(0.25 * np.pi * torch.ones(3))
    assert utils.allclose(SO3.exp(SO3.log(C_big)).mat, C_big.mat)

    C_small = SO3.exp(torch.zeros(3))
    assert utils.allclose(SO3.exp(SO3.log(C_small)).mat, C_small.mat)


def test_exp_log_batch():
    C = SO3.exp(torch.Tensor([[1, 2, 3],
                              [0, 0, 0]]))
    assert utils.allclose(SO3.exp(SO3.log(C)).mat, C.mat)


def test_perturb():
    C = SO3.exp(0.25 * np.pi * torch.ones(3))
    C_copy = copy.deepcopy(C)
    phi = torch.Tensor([0.1, 0.2, 0.3])
    C.perturb(phi)
    assert utils.allclose(
        C.as_matrix(), (SO3.exp(phi).dot(C_copy)).as_matrix())


def test_perturb_batch():
    C = SO3.exp(torch.Tensor([[1, 2, 3],
                              [4, 5, 6]]))
    C_copy1 = copy.deepcopy(C)
    C_copy2 = copy.deepcopy(C)

    phi = torch.Tensor([0.1, 0.2, 0.3])
    C_copy1.perturb(phi)
    assert utils.allclose(C_copy1.as_matrix(),
                          (SO3.exp(phi).dot(C)).as_matrix())

    phis = torch.Tensor([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6]])
    C_copy2.perturb(phis)
    assert utils.allclose(C_copy2.as_matrix(),
                          (SO3.exp(phis).dot(C)).as_matrix())


def test_normalize():
    C = SO3.exp(0.25 * np.pi * torch.ones(3))
    C.mat.add_(0.1)
    C.normalize()
    assert SO3.is_valid_matrix(C.mat).all()


def test_normalize_batch():
    C = SO3.exp(torch.Tensor([[1, 2, 3],
                              [4, 5, 6],
                              [0, 0, 0]]))
    assert (SO3.is_valid_matrix(C.mat) == torch.ByteTensor([1, 1, 1])).all()

    C.mat.add_(0.1)
    assert (SO3.is_valid_matrix(C.mat) == torch.ByteTensor([0, 0, 0])).all()

    C.normalize(inds=[0, 2])
    assert (SO3.is_valid_matrix(C.mat) == torch.ByteTensor([1, 0, 1])).all()

    C.normalize()
    assert SO3.is_valid_matrix(C.mat).all()


def test_inv():
    C = SO3.exp(0.25 * np.pi * torch.ones(3))
    assert utils.allclose(C.dot(C.inv()).mat, SO3.identity().mat)


def test_inv_batch():
    C = SO3.exp(torch.Tensor([[1, 2, 3],
                              [4, 5, 6]]))
    assert utils.allclose(C.dot(C.inv()).mat, SO3.identity(C.mat.shape[0]).mat)


def test_adjoint():
    C = SO3.exp(0.25 * np.pi * torch.ones(3))
    assert (C.adjoint() == C.mat).all()


def test_adjoint_batch():
    C = SO3.exp(torch.Tensor([[1, 2, 3],
                              [4, 5, 6]]))
    assert (C.adjoint() == C.mat).all()


def test_rotx():
    C_got = SO3.rotx(torch.Tensor([np.pi / 2]))
    C_expected = torch.Tensor([[1, 0, 0],
                               [0, 0, -1],
                               [0, 1, 0]])
    assert utils.allclose(C_got.mat, C_expected)


def test_rotx_batch():
    C_got = SO3.rotx(torch.Tensor([np.pi / 2, np.pi]))
    C_expected = torch.cat([torch.Tensor([[1, 0, 0],
                                          [0, 0, -1],
                                          [0, 1, 0]]).unsqueeze_(dim=0),
                            torch.Tensor([[1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, -1]]).unsqueeze_(dim=0)], dim=0)
    assert utils.allclose(C_got.mat, C_expected)


def test_roty():
    C_got = SO3.roty(torch.Tensor([np.pi / 2]))
    C_expected = torch.Tensor([[0, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 0]])
    assert utils.allclose(C_got.mat, C_expected)


def test_roty_batch():
    C_got = SO3.roty(torch.Tensor([np.pi / 2, np.pi]))
    C_expected = torch.cat([torch.Tensor([[0, 0, 1],
                                          [0, 1, 0],
                                          [-1, 0, 0]]).unsqueeze_(dim=0),
                            torch.Tensor([[-1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, -1]]).unsqueeze_(dim=0)], dim=0)
    assert utils.allclose(C_got.mat, C_expected)


def test_rotz():
    C_got = SO3.rotz(torch.Tensor([np.pi / 2]))
    C_expected = torch.Tensor([[0, -1, 0],
                               [1, 0, 0],
                               [0, 0, 1]])
    assert utils.allclose(C_got.mat, C_expected)


def test_rotz_batch():
    C_got = SO3.rotz(torch.Tensor([np.pi / 2, np.pi]))
    C_expected = torch.cat([torch.Tensor([[0, -1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]]).unsqueeze_(dim=0),
                            torch.Tensor([[-1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, 1]]).unsqueeze_(dim=0)], dim=0)
    assert utils.allclose(C_got.mat, C_expected)


def test_rpy():
    rpy = torch.Tensor([np.pi / 12, np.pi / 6, np.pi / 3])
    C_got = SO3.from_rpy(rpy)
    C_expected = SO3.rotz(torch.Tensor([rpy[2]])).dot(
        SO3.roty(torch.Tensor([rpy[1]])).dot(
            SO3.rotx(torch.Tensor([rpy[0]]))
        )
    )
    assert utils.allclose(C_got.mat, C_expected.mat)


def test_rpy_batch():
    rpy = torch.Tensor([[np.pi / 12, np.pi / 6, np.pi / 3],
                        [0, 0, 0]])
    C_got = SO3.from_rpy(rpy)
    C_expected = SO3.rotz(rpy[:, 2]).dot(
        SO3.roty(rpy[:, 1]).dot(
            SO3.rotx(rpy[:, 0])
        )
    )
    assert utils.allclose(C_got.mat, C_expected.mat)


def test_quaternion():
    q1 = torch.Tensor([1, 0, 0, 0])
    q2 = torch.Tensor([0, 1, 0, 0])
    q3 = torch.Tensor([0, 0, 1, 0])
    q4 = torch.Tensor([0, 0, 0, 1])
    q5 = 0.5 * torch.ones(4)
    q6 = -q5

    assert utils.allclose(SO3.from_quaternion(q1).to_quaternion(), q1)
    assert utils.allclose(SO3.from_quaternion(q2).to_quaternion(), q2)
    assert utils.allclose(SO3.from_quaternion(q3).to_quaternion(), q3)
    assert utils.allclose(SO3.from_quaternion(q4).to_quaternion(), q4)
    assert utils.allclose(SO3.from_quaternion(q5).to_quaternion(), q5)
    assert utils.allclose(SO3.from_quaternion(q5).mat,
                          SO3.from_quaternion(q6).mat)


def test_quaternion_batch():
    quats = torch.Tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0.5, 0.5, 0.5, 0.5]])

    assert utils.allclose(SO3.from_quaternion(quats).to_quaternion(), quats)
