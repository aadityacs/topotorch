"""Quad and rect element templates."""

import torch


class Quad4:
  """Quad element with 4 nodes.

  The nodes are numbered as follows:

        y                          3---c----2
          |                         \       |
          |                        d \      |b
          ---->x                      0---a--1
  """

  @property
  def dimension(self) -> int:
    return 2

  @property
  def num_nodes(self) -> int:
    return 4

  @property
  def meshio_type(self) -> str:
    return "quad"

  @property
  def num_edges(self) -> int:
    return 4

  @property
  def num_faces(self) -> int:
    return 4

  @property
  def num_nodes_per_face(self) -> int:
    return 2

  @property
  def edge_connectivity(self) -> torch.Tensor:
    """Connectivity of the edges of the element.

    Returns: An array of shape (num_edges, nodes_per_edge) containing the connectivity of
      the edges of the element. Here nodes_per_edge = 2. Hence, the first column
      contains the start node of the edge and the second column contains the end node of
      the edge. The edges are numbered so that the face normal points outwards.
    """
    return torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

  @property
  def face_connectivity(self) -> torch.Tensor:
    """Connectivity of the faces of the element.

    NOTE: For a quad element, the face is the same as the edge.

    Returns: An array of shape (num_faces, nodes_per_face) containing the connectivity of
      the faces of the element. Here nodes_per_face = 2. Hence, the first column
      contains the start node of the face and the second column contains the end node of
      the face. The edges are numbered so that the face normal points outwards.
    """
    return torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

  @staticmethod
  def diag_length(node_coords: torch.Tensor) -> float:
    """Diagonal length of the element.

    Compute the average of the main and secondary diagonals of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The average diagonal length of the element.
    """
    diag_main = torch.linalg.norm(node_coords[0, :] - node_coords[2, :])
    diag_secondary = torch.linalg.norm(node_coords[1, :] - node_coords[3, :])
    return 0.5 * (diag_main + diag_secondary)

  @staticmethod
  def elem_volume(node_coords: torch.Tensor) -> torch.Tensor:
    """Volume (area for quad in 2D; length in 1D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume (area) of the element.
    """
    # For a quad element, we can compute the area by dividing it into two triangles.
    # Using shoelace formula for each triangle and sum the areas.

    # Triangle 1: Nodes 0, 1, 2
    area1 = 0.5 * (
      node_coords[0, 0] * (node_coords[1, 1] - node_coords[2, 1])
      + node_coords[1, 0] * (node_coords[2, 1] - node_coords[0, 1])
      + node_coords[2, 0] * (node_coords[0, 1] - node_coords[1, 1])
    )

    # Triangle 2: Nodes 0, 2, 3
    area2 = 0.5 * (
      node_coords[0, 0] * (node_coords[2, 1] - node_coords[3, 1])
      + node_coords[2, 0] * (node_coords[3, 1] - node_coords[0, 1])
      + node_coords[3, 0] * (node_coords[0, 1] - node_coords[2, 1])
    )
    return torch.abs(area1) + torch.abs(area2)

  @staticmethod
  def face_normal(face_node_coords: torch.Tensor) -> torch.Tensor:
    """Normal of the faces of the element.

    Args:
      face_node_coords: Array of shape (nodes_per_face, num_dim) containing the
       coordinates of the nodes of the face. In this instance, the nodes_per_face = 2
       and num_dim is 2.

    Returns: An array of shape (num_dim,) containing the normal of a face.
    """
    edge = face_node_coords[1, :] - face_node_coords[0, :]
    normal = torch.tensor([-edge[1], edge[0]])
    normal = normal.float() / torch.linalg.norm(normal.float())
    return normal

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: torch.Tensor, node_coords: torch.Tensor
  ) -> torch.Tensor:
    """Get the isoparametric coordinate of a point in the element.
    NOTE: No checks are performed to ensure that the point is within the element.
    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1].
    """
    raise NotImplementedError("Not implemented for Quad4 element.")

  @staticmethod
  def shape_functions(
    isoparam_pt: torch.Tensor,
  ) -> torch.Tensor:
    """Compute the shape functions at the given xi, eta.

    Args:
      isoparam_pt: Array of (2,) of the isoparametric xi, and eta coordinate.

    Returns: Array of shape (4,) containing the shape functions at the given
      xi, and eta. The 4 corresponds to the number of nodes in the element.
    """
    return torch.tensor(
      [
        0.25 * (1 - isoparam_pt[0]) * (1 - isoparam_pt[1]),
        0.25 * (1 + isoparam_pt[0]) * (1 - isoparam_pt[1]),
        0.25 * (1 + isoparam_pt[0]) * (1 + isoparam_pt[1]),
        0.25 * (1 - isoparam_pt[0]) * (1 + isoparam_pt[1]),
      ]
    )

  @staticmethod
  def get_physical_coordinate(
    isoparam_coordn: torch.Tensor, node_coords: torch.Tensor
  ) -> torch.Tensor:
    """Get the physical coordinate of a point in the element.

    Args:
      isoparam_coordn: Array of shape (num_dim,) containing the isoparametric
        coordinates of the point. The values are in the range [-1, 1].
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the physical coordinate of the
      point.
    """
    shp_fn = Quad4.shape_functions(isoparam_coordn)
    return torch.einsum("n, nd -> d", shp_fn, node_coords)

  @staticmethod
  def shape_function_gradients_isoparametric(
    gauss_pt: torch.Tensor,
  ) -> torch.Tensor:
    """Compute the shape function gradients at the given xi, and eta.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinates.
        Here, 2 corresponds to the number of isoparametric dimensions.

    Returns: Array of (4, 2) arrays containing the shape function gradients at
      given xi, and eta. The 4 corresponds to the number of nodes in the element
      and 2 corresponds to the number of physical dimensions.
    """
    xis = torch.tensor([-1, 1, 1, -1])
    etas = torch.tensor([-1, -1, 1, 1])
    dN_dxi = torch.tensor(
      [
        0.25 * xis[0] * (1 - gauss_pt[1]),
        0.25 * xis[1] * (1 - gauss_pt[1]),
        0.25 * xis[2] * (1 + gauss_pt[1]),
        0.25 * xis[3] * (1 + gauss_pt[1]),
      ]
    )
    dN_deta = torch.tensor(
      [
        0.25 * (1 - gauss_pt[0]) * etas[0],
        0.25 * (1 + gauss_pt[0]) * etas[1],
        0.25 * (1 + gauss_pt[0]) * etas[2],
        0.25 * (1 - gauss_pt[0]) * etas[3],
      ]
    )
    return torch.stack((dN_dxi, dN_deta), dim=1)

  @classmethod
  def compute_jacobian_and_determinant(
    cls,
    gauss_pt: torch.Tensor,
    node_coords: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Jacobian of the element at the given xi, and eta.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinate.
      node_coords: Array of (4, 2) containing the x, and y coordinates of the nodes.
        The four corresponds to the number of nodes and 2 corresponds to the number
        of physical dimensions.

    Returns: A tuple containing the Jacobian and the its determinant:
      - Jacobians of shape (2, 2) containing the Jacobian of the element at
          the given xi, and eta. 2 corresponds to the number of physical dims.
      - Determinant of shape (1,) containing the determinant of the Jacobian.
    """
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)
    # num_(n)odes, (d)[i]m
    jac = torch.einsum("nd, ni -> di", gradN_isoparam, node_coords)
    det_jac = torch.linalg.det(jac)
    return jac, det_jac

  @classmethod
  def get_gradient_shape_function_physical(
    cls,
    gauss_pt: torch.Tensor,
    node_coords: torch.Tensor,
  ) -> torch.Tensor:
    """Compute the gradient of the shape functions at the given gauss point.

    Args:
      gauss_pt: Array of (2,) containing the isoparametric xi, and eta coordinate.
      node_coords: Array of (4, 2) containing the x, and y coordinates of the nodes.

    Returns: An array of shape (4, 2) containing the derivatives of the shape
      functions with respect to the physical coordinates. The 4 corresponds to the
      number of nodes in the element and 2 corresponds to the number of physical
      dimensions.
    """
    # num_(n)odes, (d)[i]m
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pt)  # {nd}
    jac, _ = cls.compute_jacobian_and_determinant(gauss_pt, node_coords)
    return torch.einsum("di, nd -> ni", torch.linalg.inv(jac), gradN_isoparam)


class Rect4(Quad4):
  """Rectangular element with 4 nodes.

  This class is a specific instance of the Quad4 element where the element is a
  rectangle. This is useful when we want to distinguish between a general Quad4
  element and a rectangle. The element is intended to be used with a structured
  grid mesh. While the Quad4 element can be used for both structured and
  unstructured grids, the Rect4 element is specifically for structured grids. Some of
  the computations can be optimized for a structured grid.


  The nodes are numbered as follows:

        y                            3---c---2
          |                          |       |
          |                        d |       |b
          ---->x                     0---a---1
  """

  @staticmethod
  def elem_volume(node_coords: torch.Tensor) -> torch.Tensor:
    """Volume (area for quad in 2D; length in 1D) of the element.

    Args:
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: The volume (area) of the element.
    """
    dx = torch.abs(node_coords[1, 0] - node_coords[0, 0])
    dy = torch.abs(node_coords[3, 1] - node_coords[0, 1])
    return dx * dy

  @staticmethod
  def get_isoparametric_coordinate_of_point(
    point_coordn: torch.Tensor, node_coords: torch.Tensor
  ) -> torch.Tensor:
    """Get the isoparametric coordinate of a point in the element.

    NOTE: No checks are performed to ensure that the point is within the element.

    Args:
      point_coordn: Array of shape (num_dim,) containing the coordinates of the point.
      node_coords: Array of shape (num_nodes, num_dim) containing the coordinates of the
        nodes of the element.

    Returns: Array of shape (num_dim,) containing the isoparametric coordinate of the
      point. The values are in the range [-1, 1].
    """
    dx = node_coords[1, 0] - node_coords[0, 0]
    dy = node_coords[3, 1] - node_coords[0, 1]
    xi = (2 * (point_coordn[0] - node_coords[0, 0]) / dx) - 1.0
    eta = (2 * (point_coordn[1] - node_coords[0, 1]) / dy) - 1.0
    return torch.tensor([xi, eta])
