"""Structural finite element solvers.

This module implements a finite element solver for solid mechanics problems,
capable of handling small (linear) deformations.

The governing equation is the balance of linear momentum in its strong form
(for a static problem):

    ∇ ⋅ P + B = 0

where:
  P : Cauchy stress tensor.
  B : Body force vector in the reference configuration.

The weak form of this equation states that the internal virtual work must equal
the external virtual work for any admissible virtual displacement. This leads to
the residual equation for a single element:

    R_e = F_int - F_ext = 0

where:
  F_int : Nodal internal forces, computed from the stress state within the element.
  F_ext : Nodal external forces (applied loads, body forces, etc.).
"""

from typing import Optional
import enum
import numpy as np
import torch

import topotorch.src.mesher as _mesh
import topotorch.src.material as _mat
import topotorch.src.bc as _bc
import topotorch.src.utils as _utils
import topotorch.src.solver as _solver


class DisplacementField(enum.Enum):
  """The displacment fields."""

  U = 0
  V = 1
  W = 2


class FEA:
  """Linear structural elasticity finite elements in 2D."""

  def __init__(
    self,
    mesh: _mesh.Mesh,
    mat: _mat.StructuralMaterial,
    bc: _bc.BCDict,
    fixture_penalty: float = 1e8,
  ):
    """Initialize the solver.

    Attributes:
      mesh: The mesh object that contains the mesh information.
      mat: The material properties object that contains the material properties.
      bc: The boundary conditions object that contains the boundary conditions.
      fixture_penalty: The penalty value for the fixed boundary conditions.
    """
    self.mesh, self.mat, self.bc = mesh, mat, bc

    self.D0 = _utils.to_torch(self.compute_element_stiffness_template())

    indices = np.stack((self.bc["fixed_dofs"], self.bc["fixed_dofs"]), axis=-1).T

    values = fixture_penalty * np.ones((self.bc["fixed_dofs"].shape[0]))
    self.fixed_bc_penalty_matrix = torch.sparse_coo_tensor(
      indices, values, (self.mesh.num_dofs, self.mesh.num_dofs), dtype=torch.float64
    )
    self.dirichlet_penalty_force = torch.zeros(
      mesh.num_dofs,
    ).double()
    self.dirichlet_penalty_force[self.bc["fixed_dofs"]] = (
      fixture_penalty * self.bc["dirichlet_values"]
    )
    self.node_idx = [self.mesh.iK, self.mesh.jK]

  def compute_element_stiffness_template(self) -> np.ndarray:
    """Compute the element stiffness template for a bilinear mesh of unit size.

      Precompute the expression for the element stiffness template:

                    D0 = integral([B]'[C0][B])

      Where C0 corresponds to the isotropic constitutive matrix with unity
      Young's Modulus and prescribed Poisson's ratio. [B] is the gradient of
      the bilinear shape matrix.

    Returns: stiffness template matrix of size (8,8), corresponding to the 8
      dofs of a bilinear element in 2D with 2 dofs per node.
    """
    ym = 1.0
    nu = self.mat.poissons_ratio
    k = np.array(
      [
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
      ]
    )
    return (
      ym
      / (1 - nu**2)
      * np.array(
        [
          [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
          [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
          [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
          [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
          [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
          [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
          [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
          [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
      ).T
    )

  def compute_elem_stiffness_matrix(
    self,
    youngs_modulus: torch.Tensor,
  ) -> torch.Tensor:
    """
    Args:
      youngs_modulus: Array of size (num_elems,) that contain the young's modulus
        of each of the elements for FEA.
      penal: The penalization factor for the SIMP model.
    Returns: Array of size (num_elems, 8, 8) which is the structual
      stiffness matrix of each of the bilinear quad elements. Each element has
      8 dofs corresponding to the x and y displacements of the 4 noded quad
      element.
    """
    # e - element, i - elem_nodes j - elem_nodes
    return torch.einsum("e, ij -> eij", youngs_modulus, self.D0)

  def assemble_stiffness_matrix(
    self,
    elem_stiff_mtrx: torch.Tensor,
  ) -> torch.Tensor:
    """
    Args:
      elem_stiff_mtrx: Array of size (num_elems, 8, 8) which is the structual
        stiffness matrix of each of the bilinear quad elements. Each element has
        8 dofs corresponding to the u, v displacements of the 4 noded quad
        element.
    Returns: Array of size (num_dofs, num_dofs) which is the assembled global
      stiffness matrix.
    """
    return torch.sparse_coo_tensor(
      self.node_idx,
      elem_stiff_mtrx.flatten(),
      (self.mesh.num_dofs, self.mesh.num_dofs),
    )

  def solve(
    self,
    assm_stiff_mtrx: torch.Tensor,
    addn_force: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Solve the system of Finite element equations.
    Args:
      glob_stiff_mtrx: Array of size (num_dofs, num_dofs) which is the assembled
        global stiffness matrix.
      addn_force: Additional force of size (num_dofs,) to be added to the system.

    Returns: Array of size (num_dofs,) which is the displacement of the nodes.
    """
    if addn_force is None:
      addn_force = torch.zeros_like(self.bc["force"])
    force = self.bc["force"] + addn_force + self.dirichlet_penalty_force
    net_stiff_mtrx = (assm_stiff_mtrx + self.fixed_bc_penalty_matrix).coalesce()
    u = _solver.solve(net_stiff_mtrx, force).flatten()
    return u

  def compute_compliance(
    self,
    u: torch.Tensor,
    addn_force: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Objective measure for structural performance.
    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
      addn_force: Additional force of size (num_dofs, 1) to be added to the system.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    if addn_force is None:
      addn_force = torch.zeros_like(self.bc["force"])

    return torch.dot(u.view(-1), (self.bc["force"] + addn_force).view(-1))

  def loss_function(
    self,
    youngs_modulus: torch.Tensor,
    addn_force: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Wrapper function that takes in density field and returns compliance.
    Args:
      youngs_modulus: Tensor of size (num_elems,) of the young's modulus of the
        elements.
      addn_force: Additional force of size (num_dofs, 1) to be added to the
        system. This could for instance be the force due to thermal expansion.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    elem_stiffness_mtrx = self.compute_elem_stiffness_matrix(youngs_modulus)
    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    u = self.solve(glob_stiff_mtrx, addn_force)
    return self.compute_compliance(u, addn_force), u
