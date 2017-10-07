import torch
import liegroups.torch


def test_isclose():
    tol = 1e-6
    mat = torch.Tensor([0, 1, tol, 10 * tol, 0.1 * tol])
    ans = torch.ByteTensor([1, 0, 0, 0, 1])
    assert all(liegroups.torch.isclose(mat, 0., tol=tol) == ans)


def test_allclose():
    tol = 1e-6
    mat_good = torch.Tensor([0.1 * tol, 0.01 * tol, 0, 0, 0])
    mat_bad = torch.Tensor([0, 1, tol, 10 * tol, 0.1 * tol])
    assert liegroups.torch.allclose(mat_good, 0., tol=tol)
    assert not liegroups.torch.allclose(mat_bad, 0., tol=tol)
