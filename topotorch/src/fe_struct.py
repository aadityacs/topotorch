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
import torch

import topotorch.src.mesher as _mesh
import topotorch.src.material as _mat
import topotorch.src.bc as _bc
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
  ):
    """Initialize the solver.

    Attributes:
      mesh: The mesh object that contains the mesh information.
      mat: The material properties object that contains the material properties.
      bc: The boundary conditions object that contains the boundary conditions.
    """
    self.mesh, self.mat, self.bc = mesh, mat, bc
    self.node_idx = [self.mesh.iK, self.mesh.jK]

  def _compute_elem_stiffness(
    self,
    lame_lam: torch.Tensor,
    lame_mu: torch.Tensor,
    node_coords: torch.Tensor,
  ):
    """Compute the stiffness matrix of an element.

    Args:
      lame_lambda: Scalar value of the Lame (first) parameter lambda.
      lame_mu: Scalar value of the Lame (second) parameter mu.
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.

    Returns: The stiffness matrix of the element (num_dofs_per_elem, num_dofs_per_elem).
    """
    # (g)auss, (d)(i)ms, (n)(o)des, (s)pa(t)ial
    grad_shp_fn = torch.func.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_dims=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g, n, d)

    _, det_jac = torch.func.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_dims=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,)

    const_mtrx = _mat.compute_hookean_const_matrix((lame_lam, lame_mu), num_dim=2)

    stiff = torch.einsum(
      "g, g, gns, dsit, got -> ndoi",
      self.mesh.gauss_weights,
      det_jac,
      grad_shp_fn,
      const_mtrx,
      grad_shp_fn,
    )
    return stiff.reshape((self.mesh.num_dofs_per_elem, self.mesh.num_dofs_per_elem))

  def assemble_stiffness_matrix(
    self,
    elem_stiff_mtrx: torch.Tensor,
  ) -> torch.Tensor:
    """Assemble the global stiffness matrix from element stiffness matrices.

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
    force = self.bc["force"] + addn_force
    u = _solver.solve(assm_stiff_mtrx.coalesce(), force).flatten()
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
    lame_lams: torch.Tensor,
    lame_mus: torch.Tensor,
    addn_force: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Wrapper function that takes in element material props and returns compliance.

    Args:
      lame_lams: Array of size (num_elems,) that contain the Lame (first) parameter
        lambda for each element.
      lame_mus: Array of size (num_elems,) that contain the Lame (second) param mu for
        each element.
      addn_force: Additional force of size (num_dofs, 1) to be added to the
        system. This could for instance be the force due to thermal expansion.

    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    elem_stiffness_mtrx = torch.func.vmap(
      self._compute_elem_stiffness, in_dims=(0, 0, 0)
    )(lame_lams, lame_mus, self.mesh.elem_node_coords)

    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    glob_stiff_mtrx = _bc.apply_dirichlet_bc(glob_stiff_mtrx, self.bc["fixed_dofs"])

    u = self.solve(glob_stiff_mtrx, addn_force)
    return self.compute_compliance(u, addn_force), u
