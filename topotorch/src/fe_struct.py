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

import enum
import numpy as np
import torch

import topotorch.src.mesher as _mesh
import topotorch.src.material as _mat
import topotorch.src.bc as _bc



class DisplacementField(enum.Enum):
  """The displacment fields."""

  U = 0
  V = 1
  W = 2