=================
liegroups.torch
=================

The PyTorch implementation uses torch.Tensor as the backend linear algebra library, which allows the user to on the GPU or CPU and integrate with other aspects of PyTorch.

This version provides sensible options for batching the transformations themselves, as well as anything they might operate on, and is generally agnostic to the specific Tensor type (e.g., given a torch.cuda.FloatTensor as input, the output will also be a torch.cuda.FloatTensor).

.. autoclass:: liegroups.torch.SO2
    :members: cpu, cuda

.. autoclass:: liegroups.torch.SE2
    :members: cpu, cuda

.. autoclass:: liegroups.torch.SO3
    :members: cpu, cuda

.. autoclass:: liegroups.torch.SE3
    :members: cpu, cuda
