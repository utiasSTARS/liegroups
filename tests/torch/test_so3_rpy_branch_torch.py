import itertools
import math

import pytest
import torch

from liegroups.torch import SO3


def _rotation_matrix(roll, pitch, yaw, dtype):
    # Closed form of Rz(yaw) Ry(pitch) Rx(roll), independent of SO3.
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=dtype)


def _assert_angles_and_rotation(matrix, expected, dtype):
    original = matrix.clone()
    result = SO3(matrix).to_rpy()
    tolerance = 5e-7 if dtype == torch.float32 else 1e-12
    expected = torch.tensor(expected, dtype=dtype)
    assert result.shape == expected.shape
    assert result.dtype == dtype
    assert result.device == matrix.device
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected, atol=tolerance, rtol=tolerance)
    assert torch.equal(matrix, original)

    rows = result.unsqueeze(0) if result.dim() == 1 else result
    reconstructed = torch.stack([
        _rotation_matrix(*row.tolist(), dtype=dtype) for row in rows
    ])
    target = matrix.unsqueeze(0) if matrix.dim() == 2 else matrix
    assert torch.allclose(reconstructed, target,
                          atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_preserves_mixed_branch_rows_under_permutation(dtype):
    angles = [
        (0.4, math.pi / 2, -0.2),
        (-0.3, -math.pi / 2, 0.5),
        (0.2, 0.3, -0.4),
        (-0.5, -0.2, 0.6),
    ]
    expected = [
        (0.6, math.pi / 2, 0.0),
        (0.2, -math.pi / 2, 0.0),
        angles[2],
        angles[3],
    ]
    matrices = [_rotation_matrix(*row, dtype=dtype) for row in angles]
    for order in itertools.permutations(range(len(angles))):
        _assert_angles_and_rotation(
            torch.stack([matrices[i] for i in order]),
            [expected[i] for i in order], dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_assigns_both_gimbal_lock_signs(dtype):
    angles = [(0.4, math.pi / 2, -0.2),
              (-0.3, -math.pi / 2, 0.5)]
    expected = [(0.6, math.pi / 2, 0.0),
                (0.2, -math.pi / 2, 0.0)]
    for order in [(0, 1), (1, 0)]:
        _assert_angles_and_rotation(
            torch.stack([_rotation_matrix(*angles[i], dtype=dtype)
                         for i in order]),
            [expected[i] for i in order], dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_negative_gimbal_lock_single_matrix(dtype):
    _assert_angles_and_rotation(
        _rotation_matrix(-0.3, -math.pi / 2, 0.5, dtype),
        (0.2, -math.pi / 2, 0.0), dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_positive_gimbal_lock_single_matrix(dtype):
    _assert_angles_and_rotation(
        _rotation_matrix(0.4, math.pi / 2, -0.2, dtype),
        (0.6, math.pi / 2, 0.0), dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_regular_batch_and_single_matrix(dtype):
    angles = [(0.2, 0.3, -0.4), (-0.5, -0.2, 0.6)]
    matrices = torch.stack([_rotation_matrix(*row, dtype=dtype)
                            for row in angles])
    _assert_angles_and_rotation(matrices, angles, dtype)
    _assert_angles_and_rotation(matrices[0], angles[0], dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_to_rpy_preserves_noncontiguous_mixed_input(dtype):
    angles = [(0.2, 0.3, -0.4), (0.4, math.pi / 2, -0.2),
              (-0.3, -math.pi / 2, 0.5)]
    storage = torch.zeros((3, 6, 6), dtype=dtype)
    storage[:, ::2, ::2] = torch.stack([
        _rotation_matrix(*row, dtype=dtype) for row in angles
    ])
    matrices = storage[:, ::2, ::2]
    assert not matrices.is_contiguous()
    expected = [angles[0], (0.6, math.pi / 2, 0.0),
                (0.2, -math.pi / 2, 0.0)]
    _assert_angles_and_rotation(matrices, expected, dtype)
