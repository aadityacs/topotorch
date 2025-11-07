"""Sparse linear system solver with autograd support."""

import torch
import scipy.sparse as spy_sprs
from scipy.sparse.linalg import spsolve


class SparseLinearSolve(torch.autograd.Function):
  """Custom solver to solve a sparse linear system A x = b

  The forward pass uses scipy.sparse.linalg.spsolve. The backward pass computes
    gradients w.r.t. A's non-zero values and b.
  """

  @staticmethod
  def forward(ctx, mtrx: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Forward pass: Solves A x = b.

    Args:
      ctx: Context object for saving tensors.
      mtrx: A torch sparse coo tensor of size "A" (N, N).
      b: The dense right-hand side b (N,).

    Returns: The solution x  of size (N,).
    """

    mtrx = mtrx.coalesce()

    # 1. Convert to SciPy
    mtrx_indices = mtrx.indices().detach().cpu().numpy()
    mtrx_values = mtrx.values().detach().cpu().numpy()
    mtrx_shape = mtrx.shape
    b_numpy = b.detach().cpu().numpy()

    mtrx_scipy = spy_sprs.coo_matrix(
      (mtrx_values, (mtrx_indices[0], mtrx_indices[1])), shape=mtrx_shape
    ).tocsc()

    # 2. Solve
    x_numpy = spsolve(mtrx_scipy, b_numpy)
    x_torch = torch.tensor(x_numpy, dtype=b.dtype, device=b.device)

    # --- Save for backward ---
    ctx.save_for_backward(mtrx.indices(), mtrx.values(), x_torch)
    ctx.mtrx_shape = mtrx_shape
    ctx.b_dtype = b.dtype
    ctx.b_device = b.device

    return x_torch

  @staticmethod
  def backward(ctx, grad_x):
    """Backward pass: Computes gradients dL/dA_values and dL/db.

    Args:
      ctx: Context object for retrieving saved tensors.
      grad_x (torch.Tensor): Gradient of the loss w.r.t. output x (dL/dx),
                             shape (N,).

    Returns:
      Tuple[torch.sparse_coo_tensor, torch.Tensor]:
        - Gradient w.r.t. A (sparse)
        - Gradient w.r.t. b (dense)
    """
    # Unpack saved tensors
    mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
    mtrx_shape = ctx.mtrx_shape

    # Initialize gradients as None
    grad_A_sparse = None
    grad_b = None

    # --- Reconstruct A^T in SciPy ---
    vals = mtrx_values.detach().cpu().numpy()
    idxs = (mtrx_indices[0].cpu().numpy(), mtrx_indices[1].cpu().numpy())
    mtrx_scipy = (
      spy_sprs.coo_matrix(
        (
          vals,
          idxs,
        ),
        shape=mtrx_shape,
      )
      .transpose()
      .tocsc()
    )

    # --- Solve adjoint system A^T * y = grad_x ---
    # The solution y is the gradient w.r.t. b (dL/db)
    grad_x_numpy = grad_x.cpu().numpy()

    # spsolve with 1D grad_x will return 1D grad_b
    grad_b_numpy = spsolve(mtrx_scipy, grad_x_numpy)

    grad_b_torch = torch.tensor(grad_b_numpy, dtype=ctx.b_dtype, device=ctx.b_device)

    # --- Compute gradients ---
    if ctx.needs_input_grad[1]:
      grad_b = grad_b_torch

    if ctx.needs_input_grad[0]:
      # grad_A_ij = -y_i * x_j
      # We only need the values at the non-zero indices of A
      rows, cols = mtrx_indices[0], mtrx_indices[1]

      # Efficiently gather the relevant entries from y and x
      y_at_rows = grad_b_torch.index_select(0, rows)
      x_at_cols = x_torch.index_select(0, cols)

      # Element-wise product. This replaces the sum(..., dim=1)
      grad_mtrx_values = -(y_at_rows * x_at_cols)

      # Create the sparse gradient tensor
      grad_A_sparse = torch.sparse_coo_tensor(
        mtrx_indices, grad_mtrx_values, mtrx_shape
      )

    return grad_A_sparse, grad_b


solve = SparseLinearSolve.apply
