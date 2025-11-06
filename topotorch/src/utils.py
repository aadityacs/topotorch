"""Utility functions."""

from typing import Literal, Union, Sequence
import dataclasses
import enum
import numpy as np
import scipy.special as spy_spl
import torch


class Direction(enum.Enum):
  """Euclidean Directions."""

  X = 0
  Y = 1
  Z = 2


@dataclasses.dataclass
class Extent:
  """Extent of a variable."""

  min: torch.Tensor
  max: torch.Tensor

  @property
  def range(self) -> torch.Tensor:
    return self.max - self.min

  @property
  def center(self) -> torch.Tensor:
    return 0.5 * (self.min + self.max)

  def normalize_array(self, x: torch.Tensor) -> torch.Tensor:
    """Linearly normalize `x` using `extent` ranges."""
    return (x - self.min) / self.range

  def renormalize_array(self, x: torch.Tensor) -> torch.Tensor:
    """Recover array from linearly normalized `x` using `extent` ranges."""
    return x * self.range + self.min

  def normalize_array_log(self, x: torch.Tensor) -> torch.Tensor:
    """Logarithmically normalize `x` using `extent` ranges in base 10."""
    log_min = torch.log10(self.min)
    log_range = torch.log10(self.max) - log_min
    return (torch.log10(x) - log_min) / log_range

  def renormalize_array_log(self, x: torch.Tensor) -> torch.Tensor:
    """Recover array from logarithmically normalized `x` using `extent` ranges in base 10."""
    log_min = torch.log10(self.min)
    log_range = torch.log10(self.max) - log_min
    return torch.pow(10.0, x * log_range + log_min)


def safe_power(x: torch.Tensor, exp: float) -> torch.Tensor:
  """Compute the power `x**exp` with a safe check for negative/zero values.

  This function ensures that the input `x` is positive before applying the power
    operation. If `x` is negative or zero, it returns zero. This ensures that the
  power operation does not result in undefined behavior or complex numbers.

  Args:
    x: Input array.
    exp: Exponent value.

  Returns: The result of `x**exp` if `x` is positive, otherwise zero.
  """
  z = torch.where(x <= 0.0, 1.0, x)
  return torch.where(x > 0.0, torch.pow(z, exp), 0.0)


def safe_log(x: torch.Tensor) -> torch.Tensor:
  """Compute the natural logarithm of `x` with a safe check for non-positive values.

  Args:
    x: Input array.
  Returns: The natural logarithm of `x` if `x` is positive, otherwise zero.
  """
  z = torch.where(x <= 0.0, 1.0, x)
  return torch.where(x > 0.0, torch.log(z), 0.0)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
  """Compute the square root of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    square root operation. If `x` is negative, it returns zero. This ensures that
    the square root operation does not result in undefined behavior.

  Args:
    x: Input array.
    eps: A small value to ensure numerical stability.

  Returns: The square root of `x` if `x` is non-negative, otherwise zero.
  """
  z = torch.where(x <= 0.0, 1.0, x)
  return torch.where(x >= 0.0, torch.sqrt(z), eps)


def safe_divide(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  """Compute the division of x by y with a safe check for division by zero.

  This function ensures that the denominator `y` is non-zero before applying the
    division operation. If `y` is zero, it returns zero. This ensures that the
    division operation does not result in undefined behavior.

  Args:
    x: Numerator array.
    y: Denominator array.
    eps: Small value below which the absolute value of the denominator is
      treated as zero.

  Returns: The result of `x / y` if `y` is non-zero, otherwise zero.
  """
  z = torch.where(torch.abs(y) < eps, 1.0, y)
  return torch.where(torch.abs(y) < eps, 0.0, x / z)


def safe_pnorm(x: torch.Tensor, p: float, axis: int):
  """Compute the p-norm of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    p-norm operation. If `x` is negative, it returns zero. This ensures that the
  p-norm operation does not result in undefined behavior.

  The p-norm is defined as:
               ||x||_p = (sum_i(|x_i|^p))^(1/p)

  The function is often used to compute a smooth approximation to the maximum (or minimum)
  of a set of values.

  Args:
    x: The input array.
    p: The Exponent value. The larger the value, the closer the p-norm is to the maximum.
      However, the problem becomes more nonlinear. A typical value is 6.0.
    axis: The axis along which the p-norm is computed.

  Returns: The p-norm of `x` computed in a safe manner along the specified axis.
  """
  sum_x = torch.sum(safe_power(x, p), axis=axis)
  return safe_power(sum_x, 1.0 / p)


def inverse_sigmoid(y: torch.Tensor) -> torch.Tensor:
  """The inverse of the sigmoid function.

  The sigmoid function f:x->y is defined as:

           f(x) = 1 / (1 + exp(-x))

  The inverse sigmoid function g: y->x is defined as:

           g(y) = ln(y / (1 - y))

  For details see https://tinyurl.com/y7mr76hm
  """
  return torch.log(y / (1.0 - y))


def smooth_extremum(
  x: torch.Tensor,
  order: float = 100.0,
  extreme: Literal["min", "max"] = "min",
  axis: Union[int, Sequence[int], None] = None,
) -> torch.Tensor:
  """Compute the smooth (approximate) minimum/maximum of an array.

  The function approximates the minimum/maximum of an array using the logsumexp
  function. The function is often used to compute a smooth approximation to the
  maximum (or minimum) of a set of values maintaining differentiability.

  Args:
    x: Array of whose entries we wish to compute the minimum.
    order: A float that ensures that the values are scaled appropriately to
      ensure no numerical overflow/underflow. Further, depending upon the
      magnitudes of the entry, experimenting with different values of `order`
      can result in better answers.
    extreme: Whether we wish to compute the minima or the maxima.
    axis: The axis along which the extremum is computed. If None, the extremum
      is computed over the entire array.
  """
  with torch.no_grad():
    scale = torch.amax(torch.abs(x)) / order
  sgn = -1.0 if extreme == "min" else 1.0
  if axis is None:
    return scale * sgn * torch.logsumexp(sgn * x / scale, dim=list(range(x.dim())))
  return scale * sgn * torch.logsumexp(sgn * x / scale, dim=axis)


def gauss_integ_points_weights(
  order: int,
  dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Returns the Gauss integration points and weights for the given order and
    dimension. The number of gauss points is order^dimension.

  Args:
    order (int): The order of the Gauss quadrature.
    dimension (int): The dimension of the integration. Must be in range (1, 3).

  Returns: A tuple containing the integration points and weights.
    - points (numpy.ndarray): An array of shape (order^dimension, dimension)
       containing the integration points.
    - weights (numpy.ndarray): An array of shape (order^dimension,) containing the
        integration weights.

  Raises:
      ValueError: If dimension is not in (1, 3).
  """
  # Get 1D Gauss points and weights
  x, w = spy_spl.roots_legendre(order)

  if dimension == 1:
    points = x.reshape(-1, 1)
    weights = w

  elif dimension == 2:
    # Generate 2D points and weights as tensor products of 1D points and weights
    points = torch.tensor([[x_i, x_j] for x_i in x for x_j in x])
    weights = torch.tensor([w_i * w_j for w_i in w for w_j in w])

  elif dimension == 3:
    # Generate 3D points and weights as tensor products of 1D points and weights
    points = torch.tensor([[x_i, x_j, x_k] for x_i in x for x_j in x for x_k in x])
    weights = torch.tensor([w_i * w_j * w_k for w_i in w for w_j in w for w_k in w])

  else:
    raise ValueError("Dimension must be in (1, 3)")

  return points, weights


def threshold_filter(
  density: torch.Tensor, beta: float, eta: float = 0.5
) -> torch.Tensor:
  """Threshold project the density, pushing the values towards 0/1.

  Args:
    density: Array of size (num_elems,) that are in [0,1] that contain the
      density of the elements.
    beta: Sharpness of projection (typically ~ 1-32). Larger value indicates
      a sharper projection.
    eta: Center value about which the values are projected.

  Returns: The thresholded density value array of size (num_elems,).
  """
  v1 = torch.tanh(eta * beta)
  nm = v1 + torch.tanh(beta * (density - eta))
  dnm = v1 + torch.tanh(beta * (1.0 - eta))
  return nm / dnm


class Filters(enum.Enum):
  """Enum for different types of filters."""

  LINEAR = enum.auto()
  CIRCULAR = enum.auto()
  GAUSSIAN = enum.auto()


def create_density_filter(
  coords: torch.Tensor,
  cutoff_distance: float,
  filter_type: Filters = Filters.LINEAR,
  eps: float = 1e-12,
) -> torch.Tensor:
  """Creates a density filter to smoothen out the field.

  The density filter is to ensure that the obtained density fields do not have
  checkerboard patterns. This is common in density-based topology optimization problems.

  Args:
    coords: An array of shape (num_pts, num_dim) of the coordinates of the points.
    cutoff_distance: A float, the radius beyond which the filter has zero influence.
    filter_type: A string, one of 'linear', 'circular', or 'gaussian'.
    eps: A float, small value to avoid division by zero added to the entries.

  Returns: A BCOO sparse matrix of size (num_pts, num_pts) of the filter.
  """
  num_pts = coords.shape[0]

  distances = torch.cdist(coords, coords)

  row_indices, col_indices = torch.where(distances <= cutoff_distance)
  relevant_distances = distances[row_indices, col_indices]

  if filter_type == Filters.LINEAR:
    filter_values = 1.0 - (relevant_distances / cutoff_distance)

  elif filter_type == Filters.CIRCULAR:
    filter_values = torch.sqrt(1.0 - (relevant_distances / cutoff_distance) ** 2)

  elif filter_type == Filters.GAUSSIAN:
    sigma = cutoff_distance / 3.0
    filter_values = torch.exp(-0.5 * (relevant_distances / sigma) ** 2)

  else:
    raise ValueError(f"Unsupported filter type: {filter_type.name}")

  zero = torch.zeros(num_pts, device=coords.device, dtype=coords.dtype)
  row_sums = zero.index_add_(0, row_indices, filter_values) + eps
  inv_row_sums = 1.0 / row_sums
  normalized_filter_values = filter_values * inv_row_sums[row_indices]

  return torch.sparse_coo_tensor(
    torch.stack([row_indices, col_indices]),
    normalized_filter_values,
    size=(num_pts, num_pts),
  )


def is_point_on_segment(
  start_pt: torch.Tensor, end_pt: torch.Tensor, pt: torch.Tensor, tolerance: float = 1e-9
) -> bool:
  """Checks if a point lies on a line segment with a given tolerance.

  A point is on the segment if it is both collinear with the segment's
  endpoints and lies within the axis-aligned bounding box of the segment.

  Args:
    start_pt: Array of shape (n,) of the start point of the line segment.
    end_pt: Array of shape (n,) of the end point of the line segment.
    pt: Array of shape (n,) of the point to check.
    tolerance: A small value to account for floating-point inaccuracies.

  Returns: True if the point is on the line segment, False otherwise.
  """
  # Boundedness Check
  in_bounds = torch.all(pt >= torch.minimum(start_pt, end_pt) - tolerance) and torch.all(
    pt <= torch.maximum(start_pt, end_pt) + tolerance
  )
  if not in_bounds:
    return False

  # Collinearity Check


  vec1 = end_pt - start_pt
  vec2 = pt - start_pt

  num_dims = start_pt.shape[0]
  if num_dims == 2:
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return torch.abs(cross_product) < tolerance
  elif num_dims == 3:
    cross_product = torch.cross(vec1, vec2)
    return torch.linalg.norm(cross_product) < tolerance
  else:
    raise ValueError("Only 2D and 3D points are supported.")