"""Express constrained optimization as unconstrained loss."""

import dataclasses
import enum
from typing import Union, TypeAlias, List
import torch
import torch.nn.functional as F

import topotorch.src.utils as _utils


class ConstraintTypes(enum.Enum):
  """Types of constraints in constrained optimization problems."""

  EQUALITY = 0
  INEQUALITY = 1


class LossTypes(enum.Enum):
  PENALTY = enum.auto()
  LOG_BARRIER = enum.auto()


@dataclasses.dataclass
class PenaltyParams:
  """Parameters for the penalty method.

  In the penalty method, given the objective J and a constraint g, the loss is

              L = J + alpha*g^2

  The parameters alpha starts off with a small value and is increased at each
  iteration. This makes the enforcement of the constraints more strict as the
  optimization progresses.

  Attributes:
    del_alpha: float, the increment of the penalty parameter
  """

  del_alpha: float

  def get_alpha(self, epoch: int) -> float:
    return self.del_alpha * epoch

  def __post_init__(self):
    self.loss_type = LossTypes.PENALTY


@dataclasses.dataclass
class LogBarrierParams:
  """Parameters for the log barrier method.

  In the log barrier method, given the objective J and a constraint g, the loss is

              L = { J - log(-g)/t    , if g < -1/t^2
                  { J + t*g - log(1/t^2)/t + 1/t  , otherwise}

  The parameter t starts off with a small value and is increased at each iteration.
  This makes the enforcement of the constraints more strict as the optimization
  progresses.

  Attributes:
    t0: float, the initial barrier parameter
    mu: float, the increment of the barrier parameter
  """

  t0: float
  mu: float

  def __post_init__(self):
    self.loss_type = LossTypes.LOG_BARRIER
    if self.t0 <= 0:
      raise ValueError("t0 must be positive")
    if self.mu <= 1:
      raise ValueError("mu must be greater than 1")

  def get_t(self, epoch: int) -> float:
    return self.t0 * self.mu**epoch


LossParams: TypeAlias = Union[PenaltyParams, LogBarrierParams]


def _log_barrier_update(c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  threshold = -1.0 / (t * t)
  use_log_term = c < threshold

  log_term = -_utils.safe_log(-c) / t
  linear_term = (t * c) + (2.0 * torch.log(t) / t) + (1.0 / t)
  return torch.where(use_log_term, log_term, linear_term)


_vmap_log_barrier_update = torch.vmap(_log_barrier_update, in_dims=(0, 0))


def combined_loss(
  objective: torch.Tensor,
  cons_values: torch.Tensor,
  cons_types: List[ConstraintTypes],
  loss_params: List[LossParams],
  epoch: int,
) -> torch.Tensor:
  """Compute unconstrained loss term for a constrained optimization problem.

      The constraint optimization problem is of type
                min_(x) f(x)
                s.t.    g_i(x) <= 0 , i = 1,2,...,N
                and     h_j(x) = 0 , j = 1,2,...,M

  Args:
    objective: A scalar array that is the objective of the problem.
    cons_values: Array of (num_cons,) containing the constraint values.
    cons_types: Array of (num_cons,) containing the constraint types.
    loss_params: list of size N of the loss setting parameters.
    epoch: The current epoch of the optimization.

  NOTE: All the `loss_params` must be of the same class.

  Returns: a scalar array that is the combined loss of the objective and constraints.
  """
  num_constraints = cons_values.shape[0]
  if num_constraints == 0:
    return objective

  device = cons_values.device

  cons_types_tensor = torch.tensor(
    [c.value for c in cons_types], dtype=torch.int32, device=cons_values.device
  )

  loss_type = loss_params[0].loss_type
  loss = objective

  if loss_type == LossTypes.PENALTY:
    alphas = torch.tensor([lp.get_alpha(epoch) for lp in loss_params], device=device)
    penalty_values = torch.where(
      cons_types_tensor == ConstraintTypes.EQUALITY.value,
      cons_values**2,
      F.relu(cons_values) ** 2,
    )
    constraint_loss = torch.sum(alphas * penalty_values)

  elif loss_type == LossTypes.LOG_BARRIER:
    ts = torch.tensor([lp.get_t(epoch) for lp in loss_params], device=device)
    is_inequality = cons_types_tensor == ConstraintTypes.INEQUALITY.value
    is_equality = cons_types_tensor == ConstraintTypes.EQUALITY.value
    barrier_vals_ineq = _vmap_log_barrier_update(cons_values, ts)
    barrier_vals_eq_pos = barrier_vals_ineq  # Re-use calculation for h <= 0
    barrier_vals_eq_neg = _vmap_log_barrier_update(-cons_values, ts)  # For -h <= 0

    loss_ineq = torch.sum(torch.where(is_inequality, barrier_vals_ineq, 0.0))
    loss_eq_pos = torch.sum(torch.where(is_equality, barrier_vals_eq_pos, 0.0))
    loss_eq_neg = torch.sum(torch.where(is_equality, barrier_vals_eq_neg, 0.0))

    constraint_loss = loss_ineq + loss_eq_pos + loss_eq_neg

  else:
    raise ValueError("Unsupported loss type")

  loss += constraint_loss
  return loss
