"""Containers for material data."""

import dataclasses
from typing import Optional
import torch


@dataclasses.dataclass
class ThermalMaterial:
  """Linear thermal material constants.

  Attributes:
    thermal_conductivity: The thermal conductivity of the material [W/(m*K)].
    specific_heat: Specific heat capacity of the material [J/(kg*K)].
    mass_density: Mass density of material in [kg/m^3].
    expansion_coefficient: Thermal expansion coefficient of the material [1/K].
    latent_heat_fusion: Latent heat of fusion for phase change materials [J/kg].
    melting_temperature: Melting temperature for phase change materials [K].
    cost: Cost of the material [$ / kg].
  """

  thermal_conductivity: Optional[float] = None
  specific_heat: Optional[float] = None
  mass_density: Optional[float] = None
  expansion_coefficient: Optional[float] = None
  latent_heat_fusion: Optional[float] = None
  melting_temperature: Optional[float] = None
  cost: Optional[float] = None

  @property
  def heat_capacity(self) -> float:
    """Heat capacity of the fluid [J/(m^3*K)]."""
    return self.mass_density * self.specific_heat

  @property
  def diffusivity(self) -> float:
    """Thermal diffusivity of the fluid [m^2/s]."""
    return self.thermal_conductivity / self.heat_capacity


@dataclasses.dataclass
class StructuralMaterial:
  """Linear structural material constants.

  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
    yield_strength: Yield strength of the material [Pa].
  """

  youngs_modulus: Optional[float] = None
  poissons_ratio: Optional[float] = None
  mass_density: Optional[float] = None
  yield_strength: Optional[float] = None

  @property
  def shear_modulus(self) -> float:
    return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

  @property
  def bulk_modulus(self) -> float:
    return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

  @property
  def lame_parameters(self) -> tuple[float, float]:
    """Get the Lame parameters for the material.

    Returns: The Lame parameters as a tuple (lambda, mu) for the material.
    """
    lam = (
      self.youngs_modulus
      * self.poissons_ratio
      / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))
    )
    mu = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    return lam, mu


def get_lame_parameters_from_youngs_modulus_and_poissons_ratio(
  youngs_modulus: torch.Tensor,
  poissons_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Get the Lame parameters from Young's modulus and Poisson's ratio.

  Args:
    youngs_modulus: The Young's modulus of the material [Pa].
    poissons_ratio: The Poisson's ratio of the material [-].

  Returns: The Lame parameters as a tuple (lambda, mu) for the material.
  """
  lam = (
    youngs_modulus
    * poissons_ratio
    / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
  )
  mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
  return lam, mu


def compute_hookean_const_matrix(
  lame_parameters: tuple[float, float], num_dim: int
) -> torch.Tensor:
  """Compute the PK2 stress for a Hookean material.

  This function computes the PK2 stress tensor for a Hookean material using
  the small strain approximation. The linearized material model considers the small
  strain tensor :

              epsilon = 0.5 * (grad_u + grad_u^T)
  where grad_u is the gradient of the displacement field.

  NOTE: This function uses a generic tensorial form valid for 3D, plane strain,
  or plane stress (assuming appropriate modified Lame parameters are provided).

  Args:
    lame_parameters:A tuple containing the Lame parameters (lambda, mu).

  Returns: The Second Piola-Kirchhoff stress tensor (num_dim, num_dim).
  """
  lmbda, mu = lame_parameters
  eye = torch.eye(num_dim, dtype=torch.float64)

  # D_ijkl = lambda * d_ij * d_kl + mu * (d_ik * d_jl + d_il * d_jk)
  elasticity_tensor = lmbda * torch.einsum("ij, kl -> ijkl", eye, eye) + mu * (
    torch.einsum("ik, jl -> ijkl", eye, eye) + torch.einsum("il, jk -> ijkl", eye, eye)
  )
  return elasticity_tensor
