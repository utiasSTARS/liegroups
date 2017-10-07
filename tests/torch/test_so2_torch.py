import copy

import torch
import numpy as np

from liegroups.torch import SO2, isclose, allclose


def test_identity():
    C = SO2.identity()
    assert isinstance(C, SO2)


def test_from_angle_to_angle():
    angle = torch.Tensor([np.pi / 2.])
    assert allclose(SO2.from_angle(angle).to_angle(), angle)


def test_dot():
    C = torch.Tensor([[0, -1],
                      [1, 0]])
    C2 = C.mm(C)
    assert allclose((SO2(C).dot(SO2(C))).mat, C2)


def test_wedge():
    phi = torch.Tensor([1])
    Phi = SO2.wedge(phi)
    phis = torch.Tensor([1, 2])
    Phis = SO2.wedge(phis)
    assert (Phi == -Phi.t()).all()
    assert (Phis[0, :, :] == SO2.wedge(torch.Tensor([phis[0]]))).all()
    assert (Phis[1, :, :] == SO2.wedge(torch.Tensor([phis[1]]))).all()


def test_wedge_vee():
    phi = torch.Tensor([1])
    Phi = SO2.wedge(phi)
    phis = torch.Tensor([1, 2])
    Phis = SO2.wedge(phis)
    assert (phi == SO2.vee(Phi)).all()
    assert (phis == SO2.vee(Phis)).all()


def test_exp_log():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_exp_log_zeros():
    C = SO2.exp(torch.Tensor([0]))
    assert allclose(SO2.exp(SO2.log(C)).mat, C.mat)


def test_perturb():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C_copy = copy.deepcopy(C)
    phi = torch.Tensor([0.1])
    C.perturb(phi)
    assert allclose(C.as_matrix(), (SO2.exp(phi).dot(C_copy)).as_matrix())


def test_normalize():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    C.mat.add_(0.1)
    C.normalize()
    assert SO2.is_valid_matrix(C.mat).all()


def test_inv():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert allclose(C.dot(C.inv()).mat, SO2.identity().mat)


def test_adjoint():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    assert (C.adjoint() == torch.Tensor([1.])).all()


def test_transform_vectorized():
    C = SO2.exp(torch.Tensor([np.pi / 4]))
    pt1 = torch.Tensor([1, 2])
    pt2 = torch.Tensor([4, 5])
    pts = torch.cat([pt1.unsqueeze(dim=1), pt2.unsqueeze(dim=1)], dim=1)  # 2x2
    Cpt1 = C.dot(pt1)
    Cpt2 = C.dot(pt2)
    Cpts = C.dot(pts)
    assert allclose(Cpt1, Cpts[:, 0]) and allclose(Cpt2, Cpts[:, 1])
