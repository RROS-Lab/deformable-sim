import os
import pickle
import time

import numpy as np
import pandas as pd
import warp as wp

from bayes_opt import BayesianOptimization
from tqdm import trange
from warp.sim import SemiImplicitIntegrator, Mesh, collide, load_mesh
from warp.sim.model import ModelShapeMaterials
from warp.sim.render import SimRenderer

from src.model.builder import ModelBuilderAddon
from src.model.loss import compute_chamfer_loss_warp
from src.model.package import create_package
from src.model.inside_object import create_internal_object
from src.model.suction_cup import create_suction_cup
from src.model.utils import interpolate_tool_data

random_seed = 41
np.random.seed(random_seed)


@wp.kernel
def assign_package_tri_param(
    params: wp.array(dtype=wp.float32),
    package_tri_start: wp.int32,
    package_tri_end: wp.int32,
    tri_materials: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= package_tri_start and tid < package_tri_end:
        # tri_materials = [tri_ke, tri_ka, tri_kd, tri_drag, tri_lift]
        tri_materials[tid, 0] = params[0]  # tri_ke
        tri_materials[tid, 1] = params[1]  # tri_ka
        tri_materials[tid, 2] = params[2]  # tri_kd


@wp.kernel
def assign_package_edge_param(
    params: wp.array(dtype=wp.float32),
    edge_start: wp.int32,
    edge_end: wp.int32,
    edge_bending_properties: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= edge_start and tid < edge_end:
        # edge_bending_properties = [edge_ke, edge_kd]
        edge_bending_properties[tid, 0] = params[0]  # edge_ke
        edge_bending_properties[tid, 1] = params[1]  # edge_kd


@wp.kernel
def assign_spring_stiffness(
    params: wp.array(dtype=wp.float32),
    spring_start: wp.int32,
    spring_end: wp.int32,
    spring_stiffness: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= spring_start and tid < spring_end:
        # spring_stiffness = [spring_ke, spring_kd]
        spring_stiffness[tid] = params[0]  # spring_ke


@wp.kernel
def assign_spring_damping(
    params: wp.array(dtype=wp.float32),
    spring_start: wp.int32,
    spring_end: wp.int32,
    spring_damping: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= spring_start and tid < spring_end:
        # spring_damping = [spring_ke, spring_kd]
        spring_damping[tid] = params[1]  # spring_kd


@wp.kernel
def assign_shape_param(
    params: wp.array(dtype=wp.float32),
    shape_id: wp.int32,
    shape_materials: ModelShapeMaterials,
):
    tid = wp.tid()
    if tid == shape_id:
        shape_materials.ke[tid] = params[0]
        shape_materials.kd[tid] = params[1]
        shape_materials.kf[tid] = params[2]
        shape_materials.mu[tid] = params[3]


@wp.kernel
def assign_spring_rest_length(
    spring_rest_length: wp.float32,
    spring_start: wp.int32,
    spring_end: wp.int32,
    spring_rest_lengths: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= spring_start and tid < spring_end:
        spring_rest_lengths[tid] = spring_rest_length


@wp.kernel
def assign_suction_cup_init_transform(
    particle_q: wp.array(dtype=wp.vec3),
    suction_vertex_start: wp.int32,
    tool_init_pos: wp.vec3,
    suction_cup_pos_diff: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    suction_cup_pos_diff[tid] = particle_q[tid + suction_vertex_start] - tool_init_pos


@wp.kernel
def actuate_suction_cup(
    particle_q: wp.array(dtype=wp.vec3),
    suction_vertex_start: wp.int32,
    tool_pos: wp.vec3,
    tool_orientation: wp.quat,
    tool_velocity: wp.vec3,
    suction_cup_pos_diff: wp.array(dtype=wp.vec3),
    actuator_params: wp.array(dtype=wp.float32),
    particle_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    desired_pos = wp.quat_rotate(tool_orientation, suction_cup_pos_diff[tid]) + tool_pos

    pos_error = desired_pos - particle_q[tid + suction_vertex_start]

    particle_qd[tid + suction_vertex_start] = (
        actuator_params[0] * pos_error + actuator_params[1] * tool_velocity
    )


@wp.kernel
def internal_object_loss(
    body_pos: wp.vec3,
    internal_object_pos: wp.vec3,
    internal_object_loss: wp.array(dtype=wp.float32),
):
    diff = body_pos - internal_object_pos
    internal_object_loss[0] += wp.dot(diff, diff)


@wp.kernel
def sample_particle(
    particle_q: wp.array(dtype=wp.vec3),
    package_vertex_start: wp.int32,
    package_dim_x: wp.int32,
    package_sample_dim_x: wp.int32,
    particle_in_between: wp.int32,
    sampled_particle_q: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    sample_i = tid // package_sample_dim_x
    sample_j = tid % package_sample_dim_x

    stride_x = 1 + particle_in_between
    stride_y = 1 + particle_in_between

    particle_i = sample_i * stride_y
    particle_j = sample_j * stride_x

    index = package_vertex_start + (particle_i * package_dim_x) + particle_j

    sampled_particle_q[tid] = particle_q[index]


@wp.kernel
def package_loss(
    sample_step_loss: wp.array(dtype=wp.float32),
    total_loss: wp.array(dtype=wp.float32),
):
    total_loss[0] += sample_step_loss[0]


@wp.kernel
def sum_loss(
    internal_object_loss: wp.array(dtype=wp.float32),
    package_particle_loss: wp.array(dtype=wp.float32),
    internal_object_loss_weight: wp.float32,
    package_particle_loss_weight: wp.float32,
    sim_time_length: wp.float32,
    total_loss: wp.array(dtype=wp.float32),
):
    total_loss[0] = (
        internal_object_loss[0] * internal_object_loss_weight
        + package_particle_loss[0] * package_particle_loss_weight
    ) / sim_time_length


class PackageParameterOptimEnv:
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "result",
        iterations: int = 50,
        traj_iterations: int = 3,
        train_test_ratio: float = 0.8,
        init_points: int = 5,
    ):
        if os.path.exists(
            os.path.join(
                os.getcwd(),
                data_dir,
            )
        ):
            self.data_files = [
                os.path.join(os.getcwd(), data_dir, f)
                for f in os.listdir(os.path.join(os.getcwd(), data_dir))
                if f.endswith(".pkl")
            ]
        else:
            raise FileNotFoundError(
                f"Data directory {os.path.join(os.getcwd(),data_dir)} not found."
            )
        self.env_name = "package_parameter_sampling"
        self.output_dir = os.path.join(os.getcwd(), output_dir, self.env_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.fps = 60
        self.sample_fps = 240
        self.frame_dt = 1.0 / self.fps
        self.num_substeps = 512
        self.sim_dt = self.frame_dt / self.num_substeps
        self.sim_rest_time = 2.0

        self.iterations = iterations
        self.iteration_count = 0

        self.traj_iterations = traj_iterations
        self.train_index = np.random.choice(
            len(self.data_files),
            round(train_test_ratio * len(self.data_files)),
            replace=False,
        )
        self.test_index = np.setdiff1d(
            np.arange(len(self.data_files)), self.train_index
        )

        self.init_points = init_points

        self.integrator = SemiImplicitIntegrator()

        self.minimum_total_loss = 1e2
        self.minimum_internal_object_loss = 1e4
        self.minimum_package_particle_loss = 1e4
        self.maximum_total_loss = 1e2
        self.maximum_internal_object_loss = 1e4
        self.maximum_package_particle_loss = 1e4

    def create_parameters(
        self,
        package_tri_params=wp.array(
            [2.0e5, 5.0e3, 7.0e1],
            dtype=wp.float32,
            requires_grad=True,
        ),
        package_bending_params=wp.array(
            [5.0e2, 2.0e-2],
            dtype=wp.float32,
            requires_grad=True,
        ),
        spring_material_params=wp.array(
            [3.0e6, 3.0e1],
            dtype=wp.float32,
            requires_grad=True,
        ),
        shape_params=wp.array(
            [1.0e4, 1.0e2, 10.0, 0.5],
            dtype=wp.float32,
            requires_grad=True,
        ),
        particle_in_between=3,
    ):
        # [package_tri_ke, package_tri_ka, package_tri_kd]
        self.package_tri_params = package_tri_params

        # [package_edge_ke, package_edge_kd]
        self.package_bending_params = package_bending_params

        # [suction_cup_spring_ke, suction_cup_spring_kd]
        self.spring_material_params = spring_material_params

        # [shape_ke, shape_kd, shape_kf, shape_mu]
        self.shape_params = shape_params

        # [p, d]
        self.actuator_params = wp.array(
            [5.0, 0.8],
            dtype=wp.float32,
            requires_grad=True,
        )

        self.internal_object_loss = wp.array(
            [0.0], dtype=wp.float32, requires_grad=True
        )
        self.package_particle_loss = wp.array(
            [0.0], dtype=wp.float32, requires_grad=True
        )
        self.total_loss = wp.array([0.0], dtype=wp.float32, requires_grad=True)

        self.internal_object_loss_weight = wp.float32(1.0)
        self.package_particle_loss_weight = wp.float32(0.2)

        self.particle_in_between = particle_in_between

    def create_model(self, data_file: str):
        builder = ModelBuilderAddon(up_vector=wp.vec3(0.0, 1.0, 0.0))
        builder.gravity = -4.0

        data_pickle = pickle.load(open(data_file, "rb"))

        self.sim_time_length = data_pickle["timestamp"][-1] + self.sim_rest_time
        self.sim_frames_num = (int)(self.sim_time_length / self.frame_dt)

        package_type = data_file.split("/")[-1][0].upper()
        package_particle_initial_positions = data_pickle["initial_marker_positions"]
        package_xform = wp.transform(
            p=wp.vec3(
                np.min(package_particle_initial_positions[:, 0]),
                np.mean(package_particle_initial_positions[:, 1]),
                np.min(package_particle_initial_positions[:, 2]),
            ),
            q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2),
        )
        package_mass = 2.8e-2
        package_particle_in_between = self.particle_in_between
        package = create_package(
            package_type, package_xform, package_mass, package_particle_in_between
        )
        self.package_tri_start = builder.tri_count
        self.package_edge_start = builder.edge_count
        package_vertex_indices = builder.add_package_mesh(
            pos=package.xform.p,
            rot=package.xform.q,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=package.grid_dim_sim[1] - 1,
            dim_y=package.grid_dim_sim[0] - 1,
            cell_x=package.size[1] / (package.grid_dim_sim[1] - 1),
            cell_y=package.size[0] / (package.grid_dim_sim[0] - 1),
            mass=0.8,
            radius=package.particle_radius,
            thickness=package.size[2],
            tri_ke=1.0e4,
            tri_ka=1.0e2,
            tri_kd=1.0e2,
            edge_ke=1.0e1,
            edge_kd=1.0e-1,
        )
        self.package_tri_end = builder.tri_count
        self.package_edge_end = builder.edge_count
        self.package_vertex_start = package_vertex_indices["bottom"]["start_vertex"]
        self.package_dim_x = package.grid_dim_sim[1]
        self.package_sample_dim_x = package.grid_dim_real[1]

        internal_object_type = data_file.split("/")[-1][1].upper()
        internal_object_pos = data_pickle["object_positions"][0]
        internal_object_pos[1] = np.mean(package_particle_initial_positions[:, 1])
        internal_object_xform = wp.transform(
            p=wp.vec3(internal_object_pos),
            q=wp.quat_identity(),
        )
        internal_object_mass = data_pickle["object_mass"] * 100
        internal_object = create_internal_object(
            internal_object_type, internal_object_xform, internal_object_mass
        )
        internal_object_id = builder.add_body(
            origin=internal_object.xform,
            com=wp.vec3(0.0),
            I_m=internal_object.inertia,
            m=internal_object.mass,
        )
        internal_object_mesh = load_mesh(internal_object.mesh)
        builder.add_shape_mesh(
            body=internal_object_id,
            pos=wp.vec3(0.0, -internal_object.size[1] / 2, 0.0),
            mesh=Mesh(internal_object_mesh[0], internal_object_mesh[1]),
            ke=1.0e4,
            kd=1.0e2,
            kf=1.0e1,
            mu=0.5e0,
        )
        self.internal_object_id = internal_object_id

        suction_cup_mass = 1.0e5
        suction_cup_xform = wp.transform(
            p=wp.vec3(data_pickle["initial_suction_cup_position"]),
            q=wp.quat_identity(),
        )
        suction_cup = create_suction_cup(suction_cup_xform, suction_cup_mass)
        self.spring_start = builder.spring_count
        self.suction_vertex_start = builder.particle_count
        for idx in range(4):
            angle = 2 * wp.pi * idx / 4

            cylinder_vertex_indices = builder.add_cylinder_mesh(
                pos=suction_cup.xform.p
                + wp.vec3(
                    suction_cup.size[2] * wp.sin(angle),
                    0.0,
                    suction_cup.size[2] * wp.cos(angle),
                ),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2),
                vel=wp.vec3(0.0, 0.0, 0.0),
                inner_radius=suction_cup.size[0],
                outer_radius=suction_cup.size[1],
                dim_radius=suction_cup.grid_dim_sim[0],
                dim_height=suction_cup.grid_dim_sim[1] - 1,
                mass=suction_cup.particle_mass,
                particle_radius=package.particle_radius,
            )

            builder.connect_grids_with_springs_zxy(
                package_vertex_indices,
                cylinder_vertex_indices,
                ke=1.0e4,
                kd=1.0e1,
            )
        self.spring_end = builder.spring_count
        self.suction_vertex_end = builder.particle_count

        self.spring_rest_length = wp.float32(np.min(builder.spring_rest_length))

        self.suction_cup_pos_diff = wp.zeros(
            self.suction_vertex_end - self.suction_vertex_start, dtype=wp.vec3
        )
        self.tool_init_pos = wp.vec3(data_pickle["tool_positions"][0])

        _, self.tool_positions, self.tool_orientations, self.tool_velocities = (
            interpolate_tool_data(
                data_pickle["timestamp"],
                data_pickle["tool_positions"],
                data_pickle["tool_orientations"],
                data_pickle["tool_velocity"],
                factor=(int)(self.num_substeps / (int)(240 / self.fps)),
                rest_period=self.sim_rest_time,
            )
        )
        self.internal_object_positions = data_pickle["object_positions"]

        self.marker_positions = []
        for timestamp in data_pickle["timestamp"]:
            marker_positions = data_pickle["marker_trajectories"][timestamp]
            self.marker_positions.append(wp.array(marker_positions, dtype=wp.vec3))
        self.sample_particle_q = wp.zeros(
            package.grid_dim_real[0] * package.grid_dim_real[1],
            dtype=wp.vec3,
            requires_grad=True,
        )

        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_kf = 1.0e1
        self.model.soft_contact_mu = 0.5e0

    def step(self):
        self.traj_iteration_count = 0
        self.total_losses = []
        self.internal_object_losses = []
        self.package_particle_losses = []

        selected_indices = np.random.choice(
            self.train_index, self.traj_iterations, replace=False
        )

        for idx in range(self.traj_iterations):
            self.forward(selected_indices[idx])

            if self.finished:
                print(f"Iteration: {self.iteration_count}-{self.traj_iteration_count}")
                print(
                    f"    Internal Object Loss: {self.internal_object_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                )
                print(
                    f"    Package Particle Loss: {self.package_particle_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                )
                print(f"    Total Loss: {self.total_loss.numpy()[0]}")

                self.total_losses.append(self.total_loss.numpy()[0])
                self.internal_object_losses.append(
                    self.internal_object_loss.numpy()[0]
                    / (self.sim_time_length - self.sim_rest_time)
                )
                self.package_particle_losses.append(
                    self.package_particle_loss.numpy()[0]
                    / (self.sim_time_length - self.sim_rest_time)
                )

                self.internal_object_loss.zero_()
                self.package_particle_loss.zero_()
                self.total_loss.zero_()

                self.traj_iteration_count += 1
            else:
                self.total_losses.append(self.maximum_total_loss)
                self.internal_object_losses.append(self.maximum_internal_object_loss)
                self.package_particle_losses.append(self.maximum_package_particle_loss)

        with open(os.path.join(self.output_dir, "package_parameter.csv"), "a") as f:
            writer = csv.writer(f)
            if self.iteration_count == 0:
                writer.writerow(
                    [
                        "iteration",
                        "package_tri_ke",
                        "package_tri_ka",
                        "package_tri_kd",
                        "package_edge_ke",
                        "package_edge_kd",
                        "spring_ke",
                        "spring_kd",
                        "shape_ke",
                        "shape_kd",
                        "shape_kf",
                        "shape_mu",
                        "internal_object_losses",
                        "package_particle_losses",
                        "total_losses",
                    ]
                )
            writer.writerow(
                [
                    self.iteration_count,
                    self.package_tri_params.numpy()[0],
                    self.package_tri_params.numpy()[1],
                    self.package_tri_params.numpy()[2],
                    self.package_bending_params.numpy()[0],
                    self.package_bending_params.numpy()[1],
                    self.spring_material_params.numpy()[0],
                    self.spring_material_params.numpy()[1],
                    self.shape_params.numpy()[0],
                    self.shape_params.numpy()[1],
                    self.shape_params.numpy()[2],
                    self.shape_params.numpy()[3],
                    self.internal_object_losses,
                    self.package_particle_losses,
                    self.total_losses,
                ]
            )

        self.iteration_count += 1

        return np.mean(self.total_losses)

    def forward(self, selected_indices: int, render=True):
        data_file = self.data_files[selected_indices]
        self.create_model(data_file)

        wp.launch(
            kernel=assign_package_tri_param,
            dim=self.model.tri_count,
            inputs=(
                self.package_tri_params,
                self.package_tri_start,
                self.package_tri_end,
            ),
            outputs=(self.model.tri_materials,),
        )

        wp.launch(
            kernel=assign_package_edge_param,
            dim=self.model.edge_count,
            inputs=(
                self.package_bending_params,
                self.package_edge_start,
                self.package_edge_end,
            ),
            outputs=(self.model.edge_bending_properties,),
        )

        wp.launch(
            kernel=assign_spring_stiffness,
            dim=self.model.spring_count,
            inputs=(
                self.spring_material_params,
                self.spring_start,
                self.spring_end,
            ),
            outputs=(self.model.spring_stiffness,),
        )

        wp.launch(
            kernel=assign_spring_damping,
            dim=self.model.spring_count,
            inputs=(
                self.spring_material_params,
                self.spring_start,
                self.spring_end,
            ),
            outputs=(self.model.spring_damping,),
        )

        wp.launch(
            kernel=assign_shape_param,
            dim=self.model.shape_count,
            inputs=(
                self.shape_params,
                self.internal_object_id,
            ),
            outputs=(self.model.shape_materials,),
        )

        wp.launch(
            kernel=assign_spring_rest_length,
            dim=self.model.spring_count,
            inputs=(
                self.spring_rest_length,
                self.spring_start,
                self.spring_end,
            ),
            outputs=(self.model.spring_rest_length,),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.launch(
            kernel=assign_suction_cup_init_transform,
            dim=self.suction_vertex_end - self.suction_vertex_start,
            inputs=(
                self.state_0.particle_q,
                self.suction_vertex_start,
                self.tool_init_pos,
            ),
            outputs=(self.suction_cup_pos_diff,),
        )

        self.sim_time = 0.0
        self.sim_step_count = 0
        self.sample_step_count = 0

        self.iteration_dir = os.path.join(
            self.output_dir, "sampling", f"iteration_{self.iteration_count}"
        )
        if not os.path.exists(self.iteration_dir):
            os.makedirs(self.iteration_dir)

        self.iteration_file = os.path.join(
            self.iteration_dir,
            f"{self.traj_iteration_count}_{data_file.split('/')[-1].split('.')[0]}.usd",
        )
        self.renderer = None
        if render:
            self.renderer = SimRenderer(self.model, self.iteration_file, scaling=400.0)
        self.finished = False
        self.elapsed_time = 0.0
        sim_start_time = time.time()

        with wp.ScopedTimer("simulation"):
            for frame_idx in trange(self.sim_frames_num):
                collide(self.model, self.state_0)
                for _ in range(self.num_substeps):
                    self.state_0.clear_forces()

                    if self.sim_step_count >= len(self.tool_positions):
                        self.tool_pos = wp.vec3(self.tool_positions[-1])
                        self.tool_orientation = wp.quat(self.tool_orientations[-1])
                        self.tool_velocity = wp.vec3(0.0)
                    else:
                        self.tool_pos = wp.vec3(
                            self.tool_positions[self.sim_step_count]
                        )
                        self.tool_orientation = wp.quat(
                            self.tool_orientations[self.sim_step_count]
                        )
                        self.tool_velocity = self.tool_velocities[self.sim_step_count]
                    wp.launch(
                        kernel=actuate_suction_cup,
                        dim=self.suction_vertex_end - self.suction_vertex_start,
                        inputs=(
                            self.state_0.particle_q,
                            self.suction_vertex_start,
                            self.tool_pos,
                            self.tool_orientation,
                            self.tool_velocity,
                            self.suction_cup_pos_diff,
                            self.actuator_params,
                        ),
                        outputs=(self.state_0.particle_qd,),
                    )

                    self.integrator.simulate(
                        self.model, self.state_0, self.state_1, self.sim_dt
                    )
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)

                    if (frame_idx >= self.fps * self.sim_rest_time) and (
                        self.sim_step_count
                        % (self.num_substeps / (self.sample_fps / self.fps))
                        == 0
                    ):
                        if self.sample_step_count < len(self.internal_object_positions):
                            internal_object_pos = wp.vec3(
                                self.internal_object_positions[self.sample_step_count]
                            )

                            internal_object_sim_pos = wp.vec3(
                                self.state_0.body_q.numpy()[self.internal_object_id][:3]
                            )
                            wp.launch(
                                kernel=internal_object_loss,
                                dim=1,
                                inputs=(
                                    internal_object_sim_pos,
                                    internal_object_pos,
                                ),
                                outputs=(self.internal_object_loss,),
                            ),

                            wp.launch(
                                kernel=sample_particle,
                                dim=self.sample_particle_q.shape[0],
                                inputs=(
                                    self.state_0.particle_q,
                                    self.package_vertex_start,
                                    self.package_dim_x,
                                    self.package_sample_dim_x,
                                    self.particle_in_between,
                                ),
                                outputs=(self.sample_particle_q,),
                            )

                            particle_ground_truth = self.marker_positions[
                                self.sample_step_count
                            ]

                            _, _, particle_loss = compute_chamfer_loss_warp(
                                self.sample_particle_q, particle_ground_truth
                            )
                            wp.launch(
                                kernel=package_loss,
                                dim=1,
                                inputs=(particle_loss,),
                                outputs=(self.package_particle_loss,),
                            )

                            if (
                                self.internal_object_loss.numpy()[0]
                                > self.maximum_internal_object_loss
                                or self.package_particle_loss.numpy()[0]
                                > self.maximum_package_particle_loss
                            ):
                                self.renderer.begin_frame(self.sim_time)
                                self.renderer.render(self.state_0)
                                self.renderer.end_frame()
                                self.renderer.save()
                                return

                        self.sample_step_count += 1

                    self.sim_step_count += 1

                self.sim_time += self.frame_dt

                if render:
                    self.renderer.begin_frame(self.sim_time)
                    self.renderer.render(self.state_0)
                    self.renderer.end_frame()

        sim_end_time = time.time()
        self.elapsed_time = sim_end_time - sim_start_time

        wp.launch(
            kernel=sum_loss,
            dim=1,
            inputs=(
                self.internal_object_loss,
                self.package_particle_loss,
                self.internal_object_loss_weight,
                self.package_particle_loss_weight,
                self.sim_time_length - self.sim_rest_time,
            ),
            outputs=(self.total_loss,),
        )

        if render:
            self.renderer.save()
        self.finished = True

    def bayesian_opt(self):
        param_bounds = {
            "package_tri_ke_log_scale": (0, 8),
            "package_tri_ke_digit_scale": (1.0, 9.9),
            "package_tri_ka_log_scale": (-2, 6),
            "package_tri_ka_digit_scale": (1.0, 9.9),
            "package_tri_kd_log_scale": (-2, 3),
            "package_tri_kd_digit_scale": (1.0, 9.9),
            "package_edge_ke_log_scale": (-1, 6),
            "package_edge_ke_digit_scale": (1.0, 9.9),
            "package_edge_kd_log_scale": (-4, 2),
            "package_edge_kd_digit_scale": (1.0, 9.9),
            "shape_ke_log_scale": (0, 6),
            "shape_ke_digit_scale": (1.0, 9.9),
            "shape_kd_log_scale": (-2, 3),
            "shape_kd_digit_scale": (1.0, 9.9),
            "shape_kf_log_scale": (-2, 4),
            "shape_kf_digit_scale": (1.0, 9.9),
            "shape_mu_log_scale": (-1, 1),
            "shape_mu_digit_scale": (0.0, 9.9),
        }

        def objective_function(**params):
            self.create_parameters(
                package_tri_params=wp.array(
                    [
                        np.power(10.0, params["package_tri_ke_log_scale"])
                        * params["package_tri_ke_digit_scale"],
                        np.power(10.0, params["package_tri_ka_log_scale"])
                        * params["package_tri_ka_digit_scale"],
                        np.power(10.0, params["package_tri_kd_log_scale"])
                        * params["package_tri_kd_digit_scale"],
                    ],
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                package_bending_params=wp.array(
                    [
                        np.power(10.0, params["package_edge_ke_log_scale"])
                        * params["package_edge_ke_digit_scale"],
                        np.power(10.0, params["package_edge_kd_log_scale"])
                        * params["package_edge_kd_digit_scale"],
                    ],
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                shape_params=wp.array(
                    [
                        np.power(10.0, params["shape_ke_log_scale"])
                        * params["shape_ke_digit_scale"],
                        np.power(10.0, params["shape_kd_log_scale"])
                        * params["shape_kd_digit_scale"],
                        np.power(10.0, params["shape_kf_log_scale"])
                        * params["shape_kf_digit_scale"],
                        np.power(10.0, params["shape_mu_log_scale"])
                        * params["shape_mu_digit_scale"],
                    ],
                    dtype=wp.float32,
                    requires_grad=True,
                ),
            )

            return -self.step()

        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=param_bounds,
            random_state=random_seed,
        )

        init_param = {
            "package_tri_ke_log_scale": 5,
            "package_tri_ke_digit_scale": 2.0,
            "package_tri_ka_log_scale": 3,
            "package_tri_ka_digit_scale": 5.0,
            "package_tri_kd_log_scale": 1,
            "package_tri_kd_digit_scale": 7.0,
            "package_edge_ke_log_scale": 2,
            "package_edge_ke_digit_scale": 5.0,
            "package_edge_kd_log_scale": -2,
            "package_edge_kd_digit_scale": 2.0,
            "shape_ke_log_scale": 4,
            "shape_ke_digit_scale": 1.0,
            "shape_kd_log_scale": 2,
            "shape_kd_digit_scale": 1.0,
            "shape_kf_log_scale": 1,
            "shape_kf_digit_scale": 1.0,
            "shape_mu_log_scale": 0,
            "shape_mu_digit_scale": 0.5,
        }
        optimizer.register(params=init_param, target=objective_function(**init_param))
        for _ in range(self.init_points - 1):
            perturb_param = self.perturb_params(init_param)
            optimizer.register(
                params=perturb_param, target=objective_function(**perturb_param)
            )

        optimizer.maximize(
            init_points=5,
            n_iter=self.iterations - self.init_points - 5,
        )

        print("Best parameters found:", optimizer.max["params"])

        return optimizer.max["params"]

    def perturb_params(self, params, upper=0.1, lower=0.0):
        new_params = {}
        for key, value in params.items():
            if key.endswith("digit_scale"):
                new_params[key] = value * (1 + np.random.uniform(lower, upper))
            else:
                new_params[key] = value
        return new_params

    def eval(self, input_file: str):
        if not os.path.exists(os.path.join(os.getcwd(), input_file)):
            raise FileNotFoundError(
                f"Data file {os.path.join(os.getcwd(), input_file)} not found."
            )
        self.output_dir = os.path.join(self.output_dir, "eval")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = pd.read_csv(os.path.join(os.getcwd(), input_file))
        self.train_index = str(df["train_index"][1])[1:-1].split()
        self.train_index = [int(x) for x in self.train_index]
        self.test_index = str(df["test_index"][1])[1:-1].split()
        self.test_index = [int(x) for x in self.test_index]

        package_tri_params = np.array(
            [
                df.at[df.index[-1], "package_tri_ke"],
                df.at[df.index[-1], "package_tri_ka"],
                df.at[df.index[-1], "package_tri_kd"],
            ]
        )
        package_bending_params = np.array(
            [
                df.at[df.index[-1], "package_edge_ke"],
                df.at[df.index[-1], "package_edge_kd"],
            ]
        )
        spring_material_params = np.array(
            [
                df.at[df.index[-1], "spring_ke"],
                df.at[df.index[-1], "spring_kd"],
            ],
        )
        shape_params = np.array(
            [
                df.at[df.index[-1], "shape_ke"],
                df.at[df.index[-1], "shape_kd"],
                df.at[df.index[-1], "shape_kf"],
                df.at[df.index[-1], "shape_mu"],
            ],
        )

        with open(os.path.join(self.output_dir, "evaluation.csv"), "w+") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteration",
                    "package_tri_ke",
                    "package_tri_ka",
                    "package_tri_kd",
                    "package_edge_ke",
                    "package_edge_kd",
                    "spring_ke",
                    "spring_kd",
                    "shape_ke",
                    "shape_kd",
                    "shape_kf",
                    "shape_mu",
                    "internal_object_losses",
                    "package_particle_losses",
                    "total_losses",
                ]
            )

        for perturb_level in range(6):
            self.traj_iteration_count = 0
            self.create_parameters(
                package_tri_params=wp.array(
                    package_tri_params * (1 + 0.1 * perturb_level),
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                package_bending_params=wp.array(
                    package_bending_params * (1 + 0.1 * perturb_level),
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                spring_material_params=wp.array(
                    spring_material_params * (1 + 0.1 * perturb_level),
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                shape_params=wp.array(
                    shape_params * (1 + 0.1 * perturb_level),
                    dtype=wp.float32,
                    requires_grad=True,
                ),
            )

            self.total_losses = []
            self.internal_object_losses = []
            self.package_particle_losses = []

            for train_file_idx in self.train_index:
                self.forward(train_file_idx)

                if self.finished:
                    print(
                        f"Iteration: {self.iteration_count}-{self.traj_iteration_count}"
                    )
                    print(
                        f"    Internal Object Loss: {self.internal_object_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(
                        f"    Package Particle Loss: {self.package_particle_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(f"    Total Loss: {self.total_loss.numpy()[0]}")

                    self.total_losses.append(self.total_loss.numpy()[0])
                    self.internal_object_losses.append(
                        self.internal_object_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )
                    self.package_particle_losses.append(
                        self.package_particle_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )

                    self.internal_object_loss.zero_()
                    self.package_particle_loss.zero_()
                    self.total_loss.zero_()

                self.traj_iteration_count += 1

            with open(os.path.join(self.output_dir, "evaluation.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"train_{perturb_level}",
                        self.package_tri_params.numpy()[0],
                        self.package_tri_params.numpy()[1],
                        self.package_tri_params.numpy()[2],
                        self.package_bending_params.numpy()[0],
                        self.package_bending_params.numpy()[1],
                        self.spring_material_params.numpy()[0],
                        self.spring_material_params.numpy()[1],
                        self.shape_params.numpy()[0],
                        self.shape_params.numpy()[1],
                        self.shape_params.numpy()[2],
                        self.shape_params.numpy()[3],
                        self.internal_object_losses,
                        self.package_particle_losses,
                        self.total_losses,
                    ]
                )

            self.total_losses = []
            self.internal_object_losses = []
            self.package_particle_losses = []

            for test_file_idx in self.test_index:
                self.forward(test_file_idx)

                if self.finished:
                    print(
                        f"Iteration: {self.iteration_count}-{self.traj_iteration_count}"
                    )
                    print(
                        f"    Internal Object Loss: {self.internal_object_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(
                        f"    Package Particle Loss: {self.package_particle_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(f"    Total Loss: {self.total_loss.numpy()[0]}")

                    self.total_losses.append(self.total_loss.numpy()[0])
                    self.internal_object_losses.append(
                        self.internal_object_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )
                    self.package_particle_losses.append(
                        self.package_particle_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )

                    self.internal_object_loss.zero_()
                    self.package_particle_loss.zero_()
                    self.total_loss.zero_()

                self.traj_iteration_count += 1

            with open(os.path.join(self.output_dir, "evaluation.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"test_{perturb_level}",
                        self.package_tri_params.numpy()[0],
                        self.package_tri_params.numpy()[1],
                        self.package_tri_params.numpy()[2],
                        self.package_bending_params.numpy()[0],
                        self.package_bending_params.numpy()[1],
                        self.spring_material_params.numpy()[0],
                        self.spring_material_params.numpy()[1],
                        self.shape_params.numpy()[0],
                        self.shape_params.numpy()[1],
                        self.shape_params.numpy()[2],
                        self.shape_params.numpy()[3],
                        self.internal_object_losses,
                        self.package_particle_losses,
                        self.total_losses,
                    ]
                )

            self.iteration_count += 1

    def benchmark(self, input_file: str):
        if not os.path.exists(os.path.join(os.getcwd(), input_file)):
            raise FileNotFoundError(
                f"Data file {os.path.join(os.getcwd(), input_file)} not found."
            )
        self.output_dir = os.path.join(self.output_dir, "benchmark")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = pd.read_csv(os.path.join(os.getcwd(), input_file))
        self.test_index = str(df["test_index"][1])[1:-1].split()
        self.test_index = [int(x) for x in self.test_index]

        package_tri_params = np.array(
            [
                df.at[df.index[-1], "package_tri_ke"],
                df.at[df.index[-1], "package_tri_ka"],
                df.at[df.index[-1], "package_tri_kd"],
            ]
        )
        package_bending_params = np.array(
            [
                df.at[df.index[-1], "package_edge_ke"],
                df.at[df.index[-1], "package_edge_kd"],
            ]
        )
        spring_material_params = np.array(
            [
                df.at[df.index[-1], "spring_ke"],
                df.at[df.index[-1], "spring_kd"],
            ],
        )
        shape_params = np.array(
            [
                df.at[df.index[-1], "shape_ke"],
                df.at[df.index[-1], "shape_kd"],
                df.at[df.index[-1], "shape_kf"],
                df.at[df.index[-1], "shape_mu"],
            ],
        )

        with open(os.path.join(self.output_dir, "benchmark.csv"), "w+") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "particle_in_between",
                    "package_tri_ke",
                    "package_tri_ka",
                    "package_tri_kd",
                    "package_edge_ke",
                    "package_edge_kd",
                    "spring_ke",
                    "spring_kd",
                    "shape_ke",
                    "shape_kd",
                    "shape_kf",
                    "shape_mu",
                    "internal_object_losses",
                    "package_particle_losses",
                    "total_losses",
                    "elapsed_times",
                    "sim_time_length",
                    "avg_fps",
                ]
            )

        for particle_in_between in range(1, 5):
            self.traj_iteration_count = 0
            self.create_parameters(
                package_tri_params=wp.array(
                    package_tri_params,
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                package_bending_params=wp.array(
                    package_bending_params,
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                spring_material_params=wp.array(
                    spring_material_params,
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                shape_params=wp.array(
                    shape_params,
                    dtype=wp.float32,
                    requires_grad=True,
                ),
                particle_in_between=particle_in_between,
            )

            self.total_losses = []
            self.internal_object_losses = []
            self.package_particle_losses = []
            self.elapsed_times = []
            self.sim_time_lengths = []

            for test_file_idx in self.test_index:
                self.forward(test_file_idx, render=False)

                if self.finished:
                    print(
                        f"Iteration: {self.iteration_count}-{self.traj_iteration_count}"
                    )
                    print(
                        f"    Internal Object Loss: {self.internal_object_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(
                        f"    Package Particle Loss: {self.package_particle_loss.numpy()[0] / (self.sim_time_length - self.sim_rest_time)}"
                    )
                    print(f"    Total Loss: {self.total_loss.numpy()[0]}")

                    self.total_losses.append(self.total_loss.numpy()[0])
                    self.internal_object_losses.append(
                        self.internal_object_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )
                    self.package_particle_losses.append(
                        self.package_particle_loss.numpy()[0]
                        / (self.sim_time_length - self.sim_rest_time)
                    )
                    self.elapsed_times.append(self.elapsed_time)
                    self.sim_time_lengths.append(
                        self.sim_time_length - self.sim_rest_time
                    )

                    self.internal_object_loss.zero_()
                    self.package_particle_loss.zero_()
                    self.total_loss.zero_()
                    self.elapsed_time = 0.0

                self.traj_iteration_count += 1

            with open(os.path.join(self.output_dir, "benchmark.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"test_{particle_in_between}",
                        self.package_tri_params.numpy()[0],
                        self.package_tri_params.numpy()[1],
                        self.package_tri_params.numpy()[2],
                        self.package_bending_params.numpy()[0],
                        self.package_bending_params.numpy()[1],
                        self.spring_material_params.numpy()[0],
                        self.spring_material_params.numpy()[1],
                        self.shape_params.numpy()[0],
                        self.shape_params.numpy()[1],
                        self.shape_params.numpy()[2],
                        self.shape_params.numpy()[3],
                        self.internal_object_losses,
                        self.package_particle_losses,
                        self.total_losses,
                        self.elapsed_times,
                        self.sim_time_lengths,
                        np.mean(
                            np.array(self.sim_time_lengths)
                            / np.array(self.elapsed_times)
                        ),
                    ]
                )


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for sampling.",
    )
    parser.add_argument(
        "--train_test_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to test data.",
    )
    parser.add_argument(
        "--init_points",
        type=int,
        default=5,
        help="Number of initial registration points for Bayesian optimization.",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--benchmark",
        type=bool,
        default=False,
        help="Benchmark mode.",
    )

    args = parser.parse_known_args()[0]

    env = PackageParameterOptimEnv(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        train_test_ratio=args.train_test_ratio,
        init_points=args.init_points,
    )

    if args.eval:
        env.eval("best.csv")
    elif args.benchmark:
        env.benchmark("best.csv")
    else:
        # save train and test index
        with open(os.path.join(env.output_dir, "train_test_index.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["train_index", "test_index"])
            writer.writerow([env.train_index, env.test_index])

        best_params = env.bayesian_opt()
        # save best_params
        with open(os.path.join(env.output_dir, "best_params.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            for key, value in best_params.items():
                writer.writerow([key, value])

    print("Done.")
