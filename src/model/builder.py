import warp as wp

from warp.utils import MeshAdjacency
from warp.sim.model import ModelBuilder


class ModelBuilderAddon(ModelBuilder):
    # Default particle settings
    default_particle_radius = 0.01

    # Default triangle soft mesh settings
    default_tri_ke = 100.0
    default_tri_ka = 100.0
    default_tri_kd = 10.0
    default_tri_drag = 0.0
    default_tri_lift = 0.0

    # Default distance constraint properties
    default_spring_ke = 100.0
    default_spring_kd = 0.0

    # Default edge bending properties
    default_edge_ke = 100.0
    default_edge_kd = 0.0

    # Default rigid shape contact material properties
    default_shape_ke = 1.0e5
    default_shape_kd = 1000.0
    default_shape_kf = 1000.0
    default_shape_ka = 0.0
    default_shape_mu = 0.5
    default_shape_restitution = 0.0
    default_shape_density = 1000.0
    default_shape_thickness = 1e-5

    # Default joint settings
    default_joint_limit_ke = 100.0
    default_joint_limit_kd = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_package_mesh(
        self,
        pos: wp.vec3,
        rot: wp.quat,
        vel: wp.vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        thickness: float,
        mass: float,
        radius: float = default_particle_radius,
        reverse_winding: bool = False,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
        edge_ke: float = default_edge_ke,
        edge_kd: float = default_edge_kd,
    ):
        """Helper to create a regular planar cloth grid

        Creates two rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            radius: The radius of each particle, used for collision detection
            reverse_winding: Flip the winding of the mesh
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        vertex_indices = {"top": {}, "bottom": {}}
        for layer in ["top", "bottom"]:
            vertex_indices[layer]["start_vertex"] = len(self.particle_q)
            for y in range(0, dim_y + 1):
                for x in range(0, dim_x + 1):
                    g = wp.vec3(
                        x * cell_x,
                        y * cell_y,
                        thickness / 2 if layer == "top" else -thickness / 2,
                    )
                    p = wp.quat_rotate(rot, g) + pos
                    m = mass

                    self.add_particle(p, vel, m, radius=radius)
            vertex_indices[layer]["end_vertex"] = len(self.particle_q)

        triangle_indices = {"top": {}, "bottom": {}}
        for layer in ["top", "bottom"]:
            triangle_indices[layer]["start_tri"] = len(self.tri_indices)
            start_vertex = vertex_indices[layer]["start_vertex"]
            for y in range(1, dim_y + 1):
                for x in range(1, dim_x + 1):
                    if reverse_winding:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                        )
                        tri2 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )
                    else:
                        tri1 = (
                            start_vertex + grid_index(x - 1, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )
                        tri2 = (
                            start_vertex + grid_index(x, y - 1, dim_x + 1),
                            start_vertex + grid_index(x, y, dim_x + 1),
                            start_vertex + grid_index(x - 1, y, dim_x + 1),
                        )

                    self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                    self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

            triangle_indices[layer]["end_tri"] = len(self.tri_indices)

        for layer in ["top", "bottom"]:
            start_tri = triangle_indices[layer]["start_tri"]
            end_tri = triangle_indices[layer]["end_tri"]

            adj = MeshAdjacency(
                self.tri_indices[start_tri:end_tri], end_tri - start_tri
            )

            for _k, e in adj.edges.items():
                self.add_edge(
                    e.o0, e.o1, e.v0, e.v1, edge_ke=edge_ke, edge_kd=edge_kd
                )  # opposite 0, opposite 1, vertex 0, vertex 1

        self.process_grid_edges(
            dim_x,
            dim_y,
            vertex_indices,
            tri_ke,
            tri_ka,
            tri_kd,
            tri_drag,
            tri_lift,
            edge_ke,
            edge_kd,
            reverse_winding,
        )

        vertex_indices["cell_x"] = cell_x
        vertex_indices["cell_y"] = cell_y
        vertex_indices["dim_x"] = dim_x
        vertex_indices["dim_y"] = dim_y

        return vertex_indices

    def process_edge(
        self,
        start_vertex_coords,
        dim_x,
        dim_y,
        vertex_indices,
        tri_ke,
        tri_ka,
        tri_kd,
        tri_drag,
        tri_lift,
        edge_ke,
        edge_kd,
        reverse_winding,
    ):
        """Process a single edge of the grid to create triangles and edges.

        Args:
            start_vertex_coords: tuple of (x, y, is_x_varying)
                x, y: Starting coordinates
                is_x_varying: True if x varies in the loop, False if y varies
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        start_tri = len(self.tri_indices)
        x, y, is_x_varying = start_vertex_coords

        # Determine loop range and coordinates
        dim = dim_x if is_x_varying else dim_y
        for i in range(1, dim + 1):
            curr_x = i if is_x_varying else x
            curr_y = y if is_x_varying else i
            prev_x = curr_x - 1 if is_x_varying else curr_x
            prev_y = curr_y if is_x_varying else curr_y - 1

            upper_start_vertex = vertex_indices["top"]["start_vertex"]
            lower_start_vertex = vertex_indices["bottom"]["start_vertex"]

            if reverse_winding:
                tri1 = (
                    lower_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                    lower_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                    upper_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                )
                tri2 = (
                    lower_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                    upper_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                    upper_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                )
            else:
                tri1 = (
                    lower_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                    lower_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                    upper_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                )
                tri2 = (
                    lower_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                    upper_start_vertex + grid_index(curr_x, curr_y, dim_x + 1),
                    upper_start_vertex + grid_index(prev_x, prev_y, dim_x + 1),
                )

            self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        end_tri = len(self.tri_indices)

        # Process edges
        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        for _k, e in adj.edges.items():
            self.add_edge(e.o0, e.o1, e.v0, e.v1, edge_ke=edge_ke, edge_kd=edge_kd)

    def process_grid_edges(
        self,
        dim_x,
        dim_y,
        vertex_indices,
        tri_ke,
        tri_ka,
        tri_kd,
        tri_drag,
        tri_lift,
        edge_ke,
        edge_kd,
        reverse_winding,
    ):
        """Process all four edges of the grid."""
        # Define the four edges: (x, y, is_x_varying)
        edges = [
            (1, 0, True),  # Bottom edge
            (0, 1, False),  # Left edge
            (1, dim_y, True),  # Top edge
            (dim_x, 1, False),  # Right edge
        ]

        for edge in edges:
            self.process_edge(
                edge,
                dim_x,
                dim_y,
                vertex_indices,
                tri_ke,
                tri_ka,
                tri_kd,
                tri_drag,
                tri_lift,
                edge_ke,
                edge_kd,
                reverse_winding,
            )

    def add_cylinder_mesh(
        self,
        pos: wp.vec3,
        rot: wp.quat,
        vel: wp.vec3,
        inner_radius: float,
        outer_radius: float,
        dim_radius: int,
        dim_height: int,
        mass: float,
        particle_radius: float = default_particle_radius,
        reverse_winding: bool = False,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
        edge_ke: float = default_edge_ke,
        edge_kd: float = default_edge_kd,
    ):
        def grid_index(r, h, dim_radius):
            return h * dim_radius + r

        start_vertex = len(self.particle_q)
        for h in range(0, dim_height + 1):
            for r in range(0, dim_radius):
                theta = r * 2 * wp.pi / dim_radius
                radius = inner_radius + (outer_radius - inner_radius) * h / (
                    dim_height + 1
                )

                g = wp.vec3(
                    radius * wp.cos(theta),
                    radius * wp.sin(theta),
                    0,
                )
                p = wp.quat_rotate(rot, g) + pos
                m = mass

                self.add_particle(p, vel, m, radius=particle_radius)
        end_vertex = len(self.particle_q)

        start_tri = len(self.tri_indices)
        for h in range(1, dim_height + 1):
            for r in range(1, dim_radius):
                if reverse_winding:
                    tri1 = (
                        start_vertex + grid_index(r - 1, h - 1, dim_radius),
                        start_vertex + grid_index(r, h - 1, dim_radius),
                        start_vertex + grid_index(r, h, dim_radius),
                    )
                    tri2 = (
                        start_vertex + grid_index(r - 1, h - 1, dim_radius),
                        start_vertex + grid_index(r, h, dim_radius),
                        start_vertex + grid_index(r - 1, h, dim_radius),
                    )
                else:
                    tri1 = (
                        start_vertex + grid_index(r - 1, h - 1, dim_radius),
                        start_vertex + grid_index(r, h - 1, dim_radius),
                        start_vertex + grid_index(r - 1, h, dim_radius),
                    )
                    tri2 = (
                        start_vertex + grid_index(r, h - 1, dim_radius),
                        start_vertex + grid_index(r, h, dim_radius),
                        start_vertex + grid_index(r - 1, h, dim_radius),
                    )

                self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
                self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        for h in range(1, dim_height + 1):
            if reverse_winding:
                tri1 = (
                    start_vertex + grid_index(dim_radius - 1, h - 1, dim_radius),
                    start_vertex + grid_index(0, h - 1, dim_radius),
                    start_vertex + grid_index(0, h, dim_radius),
                )
                tri2 = (
                    start_vertex + grid_index(dim_radius - 1, h - 1, dim_radius),
                    start_vertex + grid_index(0, h, dim_radius),
                    start_vertex + grid_index(dim_radius - 1, h, dim_radius),
                )
            else:
                tri1 = (
                    start_vertex + grid_index(dim_radius - 1, h - 1, dim_radius),
                    start_vertex + grid_index(0, h - 1, dim_radius),
                    start_vertex + grid_index(dim_radius - 1, h, dim_radius),
                )
                tri2 = (
                    start_vertex + grid_index(0, h - 1, dim_radius),
                    start_vertex + grid_index(0, h, dim_radius),
                    start_vertex + grid_index(dim_radius - 1, h, dim_radius),
                )

            self.add_triangle(*tri1, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            self.add_triangle(*tri2, tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        end_tri = len(self.tri_indices)

        adj = MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)
        for _k, e in adj.edges.items():
            self.add_edge(e.o0, e.o1, e.v0, e.v1, edge_ke=edge_ke, edge_kd=edge_kd)

        return {"start_vertex": start_vertex, "end_vertex": end_vertex}

    def connect_grids_with_springs(
        self,
        package_vertex_indices,
        ring_vertex_indices,
        ke=default_spring_ke,
        kd=default_spring_kd,
        control=0.0,
    ) -> list[int]:
        rect_start = package_vertex_indices["top"]["start_vertex"]
        cell_x = package_vertex_indices["cell_x"]
        cell_y = package_vertex_indices["cell_y"]
        dim_x = package_vertex_indices["dim_x"]
        dim_y = package_vertex_indices["dim_y"]

        ring_start = ring_vertex_indices["start_vertex"]
        ring_end = ring_vertex_indices["end_vertex"]

        connected_rect_indices = set()
        for radial_idx in range(ring_start, ring_end):
            radial_pos = self.particle_q[radial_idx]

            # Estimate grid position
            est_x = int((radial_pos[0] - self.particle_q[rect_start][0]) / cell_x)
            est_y = int((radial_pos[1] - self.particle_q[rect_start][1]) / cell_y)

            # Define local search range (±2 cells)
            x_start = max(0, est_x - 2)
            x_end = min(dim_x + 1, est_x + 3)
            y_start = max(0, est_y - 2)
            y_end = min(dim_y + 1, est_y + 3)

            min_dist = float("inf")
            closest_rect_idx = None

            # Search in local region
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    rect_idx = rect_start + y * (dim_x + 1) + x
                    rect_pos = self.particle_q[rect_idx]
                    dist = wp.length(rect_pos - radial_pos)

                    if dist < min_dist:
                        min_dist = dist
                        closest_rect_idx = rect_idx

            if closest_rect_idx is not None:
                self.add_spring(
                    radial_idx, closest_rect_idx, ke=ke, kd=kd, control=control
                )

            connected_rect_indices.add(closest_rect_idx)

        return connected_rect_indices

    def connect_grids_with_springs_zxy(
        self,
        package_vertex_indices,
        ring_vertex_indices,
        ke=default_spring_ke,
        kd=default_spring_kd,
        control=0.0,
    ):
        rect_start = package_vertex_indices["bottom"]["start_vertex"]
        dim_x = package_vertex_indices["dim_x"]
        dim_y = package_vertex_indices["dim_y"]
        ring_start = ring_vertex_indices["start_vertex"]
        ring_end = ring_vertex_indices["end_vertex"]
        
        connected_rect_indices = set()
        
        # Calculate the total number of vertices in the bottom layer
        total_bottom_vertices = (dim_x + 1) * (dim_y + 1)
        
        for radial_idx in range(ring_start, ring_end):
            radial_pos = self.particle_q[radial_idx]
            
            min_dist = float("inf")
            closest_rect_idx = None
            
            # Thorough search through ALL bottom layer vertices
            for offset in range(total_bottom_vertices):
                rect_idx = rect_start + offset
                rect_pos = self.particle_q[rect_idx]
                
                # Calculate distance between points
                dist = wp.length(rect_pos - radial_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_rect_idx = rect_idx
            
            if closest_rect_idx is not None:
                self.add_spring(
                    radial_idx, closest_rect_idx, ke=ke, kd=kd, control=control
                )
                connected_rect_indices.add(closest_rect_idx)
        
        return connected_rect_indices

    def add_package_grid(
        self,
        pos: wp.vec3,
        rot: wp.quat,
        vel: wp.vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        radius: float,
        thickness: float,
        package_mass: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ):

        mass = package_mass / (2 * (cell_x + 1) * (cell_y + 1) * (cell_z + 1))
        vertex_indices = {"top": {}, "bottom": {}}

        for layer in ["top", "bottom"]:
            vertex_indices[layer]["start_vertex"] = len(self.particle_q)
            for z in range(dim_z + 1):
                for y in range(dim_y + 1):
                    for x in range(dim_x + 1):
                        v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                        m = mass

                        p = (
                            wp.quat_rotate(rot, v)
                            + pos
                            + wp.vec3(0, 0, thickness if layer == "top" else -thickness)
                        )
                        self.add_particle(p, vel, m, radius=radius)
            vertex_indices[layer]["end_vertex"] = len(self.particle_q)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for layer in ["top", "bottom"]:
            start_vertex = vertex_indices[layer]["start_vertex"]
            faces.clear()
            for z in range(dim_z):
                for y in range(dim_y):
                    for x in range(dim_x):
                        v0 = grid_index(x, y, z) + start_vertex
                        v1 = grid_index(x + 1, y, z) + start_vertex
                        v2 = grid_index(x + 1, y, z + 1) + start_vertex
                        v3 = grid_index(x, y, z + 1) + start_vertex
                        v4 = grid_index(x, y + 1, z) + start_vertex
                        v5 = grid_index(x + 1, y + 1, z) + start_vertex
                        v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                        v7 = grid_index(x, y + 1, z + 1) + start_vertex

                        if (x & 1) ^ (y & 1) ^ (z & 1):
                            add_tet(v0, v1, v4, v3)
                            add_tet(v2, v3, v6, v1)
                            add_tet(v5, v4, v1, v6)
                            add_tet(v7, v6, v3, v4)
                            add_tet(v4, v1, v6, v3)

                        else:
                            add_tet(v1, v2, v5, v0)
                            add_tet(v3, v0, v7, v2)
                            add_tet(v4, v7, v0, v5)
                            add_tet(v6, v5, v2, v7)
                            add_tet(v5, v2, v7, v0)

            # add triangles
            for _k, v in faces.items():
                self.add_triangle(
                    v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift
                )

        faces.clear()
        for y in range(dim_y):
            for x in [0, dim_x - 1]:
                bottom_start_index = vertex_indices["bottom"]["start_vertex"]
                top_start_index = vertex_indices["top"]["start_vertex"]

                v0 = grid_index(x, y, dim_z) + bottom_start_index
                v1 = grid_index(x + 1, y, dim_z) + bottom_start_index
                v2 = grid_index(x + 1, y, 0) + top_start_index
                v3 = grid_index(x, y, 0) + top_start_index
                v4 = grid_index(x, y + 1, dim_z) + bottom_start_index
                v5 = grid_index(x + 1, y + 1, dim_z) + bottom_start_index
                v6 = grid_index(x + 1, y + 1, 0) + top_start_index
                v7 = grid_index(x, y + 1, 0) + top_start_index

                if (x & 1) ^ (y & 1):
                    add_tet(v0, v1, v4, v3)
                    add_tet(v2, v3, v6, v1)
                    add_tet(v5, v4, v1, v6)
                    add_tet(v7, v6, v3, v4)
                    add_tet(v4, v1, v6, v3)
                else:
                    add_tet(v1, v2, v5, v0)
                    add_tet(v3, v0, v7, v2)
                    add_tet(v4, v7, v0, v5)
                    add_tet(v6, v5, v2, v7)
                    add_tet(v5, v2, v7, v0)

        for x in range(1, dim_x - 1):
            for y in [0, dim_y - 1]:
                bottom_start_index = vertex_indices["bottom"]["start_vertex"]
                top_start_index = vertex_indices["top"]["start_vertex"]

                v0 = grid_index(x, y, dim_z) + bottom_start_index
                v1 = grid_index(x + 1, y, dim_z) + bottom_start_index
                v2 = grid_index(x + 1, y, 0) + top_start_index
                v3 = grid_index(x, y, 0) + top_start_index
                v4 = grid_index(x, y + 1, dim_z) + bottom_start_index
                v5 = grid_index(x + 1, y + 1, dim_z) + bottom_start_index
                v6 = grid_index(x + 1, y + 1, 0) + top_start_index
                v7 = grid_index(x, y + 1, 0) + top_start_index

                if (x & 1) ^ (y & 1):
                    add_tet(v0, v1, v4, v3)
                    add_tet(v2, v3, v6, v1)
                    add_tet(v5, v4, v1, v6)
                    add_tet(v7, v6, v3, v4)
                    add_tet(v4, v1, v6, v3)
                else:
                    add_tet(v1, v2, v5, v0)
                    add_tet(v3, v0, v7, v2)
                    add_tet(v4, v7, v0, v5)
                    add_tet(v6, v5, v2, v7)
                    add_tet(v5, v2, v7, v0)

        for _k, v in faces.items():
            self.add_triangle(
                v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift
            )
