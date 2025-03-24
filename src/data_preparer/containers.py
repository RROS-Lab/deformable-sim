##Contains container to store the data values
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import binned_statistic_2d
from scipy.signal import savgol_filter, medfilt
import os
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.data_preparer.utils import (
    plot_trajectory,
    plot_velocity,
    plot_acceleration,
    butter_lowpass_filter,
    plot_frame,
)

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Marker:

    def __init__(self, marker_id, package_id):

        self.marker_id = marker_id

        self.package_id = package_id
        self.initial_position = {"x": 0, "y": 0, "z": 0}
        self.neighbor_ids = []

        # Storing the trajectory, velocity and acceleration data
        self.timestamps = []
        self.marker_positions = []
        self.marker_velocities = []
        self.marker_accelerations = []

        self.start_of_motion = 0  # secs
        self.end_of_motion = 0  # secs

        return

    def set_timestamped_trajectory(self, timestamps, timestamped_trajectory):
        self.timestamps = timestamps
        self.marker_positions = timestamped_trajectory
        return

    def compute_marker_velocities(self):
        assert len(self.marker_positions) > 0, "Trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert len(self.marker_positions) == len(
            self.timestamps
        ), "Timestamps and trajectory are not of the same length"

        dt = self.timestamps

        # Computing Linear Velocity
        linear_velocity_vx = np.reshape(
            np.gradient(self.marker_positions[:, 0], dt), (-1, 1)
        )
        linear_velocity_vy = np.reshape(
            np.gradient(self.marker_positions[:, 1], dt), (-1, 1)
        )
        linear_velocity_vz = np.reshape(
            np.gradient(self.marker_positions[:, 2], dt), (-1, 1)
        )

        self.marker_velocities = np.hstack(
            (linear_velocity_vx, linear_velocity_vy, linear_velocity_vz)
        )
        # [m/s]
        # Filtering the linear velocity
        self.marker_velocities = butter_lowpass_filter(
            self.marker_velocities, cutoff=6, fs=50, order=3
        )

        return

    def compute_marker_accelerations(self):
        assert len(self.marker_positions) > 0, "Trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert len(self.marker_velocities) == 0, "Velocities are not computed"

        linear_acceleration = np.gradient(
            self.marker_velocities, self.timestamps, axis=0
        )
        # Filtering the linear acceleration
        linear_acceleration = butter_lowpass_filter(
            linear_acceleration, cutoff=6, fs=50, order=3
        )

        return

    def filter_marker_trajectory_on_start_end_time(self):
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert self.start_of_motion != 0, "Start of motion is not set"
        assert self.end_of_motion != 0, "End of motion is not set"

        # Filtering the timestamps

        if self.start_of_motion > self.timestamps[-1]:
            return False

        end_of_motion = self.end_of_motion
        if self.end_of_motion > self.timestamps[-1]:
            end_of_motion = self.timestamps[-1]

        start_idx = np.argmin(np.abs(self.timestamps - self.start_of_motion))
        end_idx = np.argmin(np.abs(self.timestamps - end_of_motion))

        if start_idx == end_idx:
            return False
        if end_idx > len(self.marker_positions):
            end_idx = len(self.marker_positions)

        self.marker_positions = self.marker_positions[start_idx:end_idx]
        self.timestamps = self.timestamps[start_idx:end_idx]
        self.timestamps = self.timestamps - self.timestamps[0]

        self.marker_velocities = self.marker_velocities[start_idx:end_idx]
        self.marker_accelerations = self.marker_accelerations[start_idx:end_idx]

        self.start_of_motion = self.timestamps[0]
        self.end_of_motion = self.timestamps[-1]

        return True

    def visualize_trajectory(self):

        plot_trajectory(
            self.marker_positions,
            None,
            self.timestamps,
            title="Marker Trajectory",
            is_2D=False,
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "marker_trajectory.png"
            ),
        )

        return

    def visualize_trajectory_2D(self):
        plot_trajectory(
            self.marker_positions,
            None,
            self.timestamps,
            title="Marker Trajectory",
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            is_2D=True,
            save_path=None,
        )

        return

    def visualize_velocities(self):
        plot_velocity(
            self.marker_velocities,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            title="Marker Velocity",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "marker_velocity.png"
            ),
        )

        return

    def visualize_acceleration(self):
        plot_acceleration(
            self.marker_accelerations,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            title="Marker Acceleration",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "marker_acceleration.png"
            ),
        )

        return


class Package:

    def __init__(self, package_type, package_dims):

        self.package_type = package_type
        self.package_dims = package_dims
        self.markers = {}

        self.start_of_motion = 0  # secs
        self.end_of_motion = 30  # secs

        self.num_markers = 0

        return

    def add_marker(self, marker: Marker):

        if marker.marker_id in self.markers:
            print(f"Marker {marker.marker_id} already exists")
        else:
            self.markers[self.num_markers] = marker

        self.num_markers += 1

    def reassign_marker_ids(self):

        return

    def visualize_package_motion(
        self,
        start_idx: int,
        timestamps: np.ndarray,  # Array of timestamps to animate
        suction_tool_tf_list: list = None,
        object_tf_list: list = None,
        world_tf_list: list = None,
        saving_filename: str = "package_motion.mp4",
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])
        all_positions = []
        for marker in self.markers.values():
            all_positions.append(marker.marker_positions)

        all_positions = np.vstack(all_positions)
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

        # Set fixed axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        def update(frame):
            ax.clear()  # Clear previous frame
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])

            timestamp = timestamps[frame]
            xyz_positions = []

            # Finding the closest timestamp
            for marker_id, marker in self.markers.items():
                # marker = self.markers[marker_id]
                if marker.marker_positions.shape[0] <= start_idx + frame:
                    continue
                xyz_positions.append(
                    [
                        marker.marker_positions[start_idx + frame, 0],
                        marker.marker_positions[start_idx + frame, 1],
                        marker.marker_positions[start_idx + frame, 2],
                    ]
                )

            xyz_positions = np.array(xyz_positions)
            if xyz_positions.shape[0] == 0:
                print(f"No data points at timestamp {timestamp}")
                return

            # Plot markers
            ax.scatter(
                xyz_positions[:, 0],
                xyz_positions[:, 1],
                xyz_positions[:, 2],
                color="b",
                label="Markers",
            )

            # Plot marker IDs
            for i, (x, y, z) in enumerate(xyz_positions):
                ax.text(x, y, z, str(i), color="black", fontsize=8)

            # Plot frames
            if suction_tool_tf_list is not None:
                suction_tool_tf = suction_tool_tf_list[frame]
                plot_frame(
                    ax,
                    suction_tool_tf[:3],
                    suction_tool_tf[3:],
                    label="Suction Tool Frame",
                )

            if object_tf_list is not None:
                object_tf = object_tf_list[frame]
                plot_frame(ax, object_tf[:3], object_tf[3:], label="Object Frame")

            if world_tf_list is None:
                world_tf = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion

            plot_frame(ax, world_tf[:3], world_tf[3:], label="World Frame")

            ax.legend()

        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(timestamps), interval=100
        )  # Adjust interval as needed
        ani.save(os.path.join(os.path.dirname(MAIN_DIR), "plots", saving_filename))
        # Show animation
        plt.show()

        return

    def filter_package_trajectory_on_start_end_time(self):
        assert len(self.markers) > 0, "Markers are empty"
        assert self.start_of_motion != 0, "Start of motion is not set"
        assert self.end_of_motion != 0, "End of motion is not set"
        invalid_markers = []
        for marker_id, marker in self.markers.items():
            marker_valid = marker.filter_marker_trajectory_on_start_end_time()
            if not marker_valid:
                invalid_markers.append(marker_id)
        for marker_id in invalid_markers:
            del self.markers[marker_id]

        self.num_markers = len(self.markers)
        if self.num_markers == 0:
            print("No markers left after filtering")

        return

    def visualize_package_at_timestamp(
        self,
        timestamp: float,
        all_timestamps: np.ndarray,
        suction_tool_tf: np.ndarray,
        object_tf: np.ndarray,
        world_tf: np.ndarray,
    ):

        xyz_positions = []
        # Finding the closest timestamp
        closes_timestamp_idx = np.argmin(np.abs(all_timestamps - timestamp))

        for marker_id in self.markers:
            marker: Marker = self.markers[marker_id]
            if marker.marker_positions.shape[0] <= closes_timestamp_idx:
                print(f"Marker {marker_id} has less data points")
                continue
            xyz_positions.append(
                [
                    marker.marker_positions[closes_timestamp_idx, 0],
                    marker.marker_positions[closes_timestamp_idx, 1],
                    marker.marker_positions[closes_timestamp_idx, 2],
                ]
            )

        xyz_positions = np.array(xyz_positions)

        ##Creating a scatter plot of the markers
        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            xyz_positions[:, 0],
            xyz_positions[:, 1],
            xyz_positions[:, 2],
            color="b",
            label="Markers",
        )

        for i, (x, y, z) in enumerate(xyz_positions):
            ax.text(x, y, z, str(i), color="black", fontsize=8)

        # Plot frames
        if suction_tool_tf is not None:
            plot_frame(
                ax,
                suction_tool_tf[:3],
                suction_tool_tf[3:],
                label=" Suction Tool Frame",
            )

        if object_tf is not None:
            plot_frame(
                ax,
                object_tf[:3],
                object_tf[3:],
                label=" Suction Tool Frame",
            )

        if world_tf is None:
            world_tf = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion

        plot_frame(ax, world_tf[:3], world_tf[3:], label="World Frame")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Show legend
        ax.legend()
        plt.show()
        return


class Tool:
    def __init__(self, tool_name: str = "four_suction_cup_tool"):
        self.tool_name = tool_name

        self.tool_positions = []
        self.tool_orientations = []
        self.tool_velocity = []
        self.tool_acceleration = []

        self.suction_cup_ids = []  # IDs for the four suction cups
        self.suction_cup_deformations = (
            []
        )  # Deformations for the four suction cups in vertical direction

        self.suction_cup_orientations = (
            []
        )  # 2D orientations (pitch and yaw) for the four suction cups

        ##Time corresponding to start of motion
        self.start_of_motion = 0  # secs
        self.end_of_motion = 0  # secs

        return

    def initialize_tool_trajectory(self, timestamps, tool_positions, tool_orientations):
        self.tool_positions = tool_positions
        self.tool_orientations = tool_orientations
        self.timestamps = timestamps
        self.end_of_motion = timestamps[-1]

        return

    def filter_tool_trajectory_on_start_end_time(
        self,
    ):
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert self.start_of_motion != 0, "Start of motion is not set"
        assert self.end_of_motion != 0, "End of motion is not set"

        # Filtering the timestamps
        start_idx = np.argmin(np.abs(self.timestamps - self.start_of_motion))
        end_idx = np.argmin(np.abs(self.timestamps - self.end_of_motion))

        self.tool_positions = self.tool_positions[start_idx:end_idx]
        self.tool_orientations = self.tool_orientations[start_idx:end_idx]
        self.timestamps = self.timestamps[start_idx:end_idx]
        self.timestamps = self.timestamps - self.timestamps[0]

        self.tool_velocity = self.tool_velocity[start_idx:end_idx]
        self.tool_acceleration = self.tool_acceleration[start_idx:end_idx]

        self.start_of_motion = self.timestamps[0]
        self.end_of_motion = self.timestamps[-1]

        return

    def compute_tool_velocity(self):
        assert len(self.tool_positions) > 0, "Tool trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert len(self.tool_orientations) == len(
            self.tool_positions
        ), "Timestamps and tool trajectory are not of the same length"

        # dt = np.diff(self.timestamps)[:, None]
        dt = self.timestamps
        # Computng Linear Velocity
        linear_velocity_vx = np.reshape(
            np.gradient(self.tool_positions[:, 0], dt), (-1, 1)
        )
        linear_velocity_vy = np.reshape(
            np.gradient(self.tool_positions[:, 1], dt), (-1, 1)
        )
        linear_velocity_vz = np.reshape(
            np.gradient(self.tool_positions[:, 2], dt), (-1, 1)
        )

        ##Computing Angular Velocity
        dq_dt = np.gradient(self.tool_orientations, self.timestamps, axis=0)
        angular_velocity = 2 * R.from_quat(self.tool_orientations).inv().apply(
            dq_dt[:, :3]
        )
        # Filterning the angular velocity
        angular_velocity = butter_lowpass_filter(
            angular_velocity, cutoff=5, fs=50, order=3
        )

        ##Alternate method to compute angular velocity
        # dt = np.diff(self.timestamps)[:, None]
        # rotations = R.from_quat(self.tool_orientations)
        # orientation_diff = (rotations[:-1].inv() * rotations[1:]).as_quat()
        # angular_velocity = 2 * orientation_diff[:, :3] / dt

        self.tool_velocity = np.hstack(
            (
                linear_velocity_vx,
                linear_velocity_vy,
                linear_velocity_vz,
                angular_velocity,
            )
        )  # [m/s, rad/s]

        return

    def compute_tool_acceleration(self):
        assert len(self.tool_positions) > 0, "Tool trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert len(self.tool_orientations) == len(
            self.tool_positions
        ), "Timestamps and tool trajectory are not of the same length"

        linear_acceleration = np.gradient(
            self.tool_velocity[:, :3], self.timestamps, axis=0
        )
        # Filtering the linear acceleration
        linear_acceleration = butter_lowpass_filter(
            linear_acceleration, cutoff=5, fs=50, order=3
        )

        # Computing Angular Acceleration
        angular_acceleration = np.gradient(
            self.tool_velocity[:, 3:], self.timestamps, axis=0
        )
        # Filtering the angular acceleration
        angular_acceleration = butter_lowpass_filter(
            angular_acceleration, cutoff=5, fs=50, order=3
        )

        self.tool_acceleration = np.hstack(
            (linear_acceleration, angular_acceleration)
        )  # [m/s^2, rad/s^2]

        return

    def compute_suction_cup_deformations(self):
        raise NotImplementedError("This method is not implemented yet")

    def compute_suction_cup_orientations(self):
        raise NotImplementedError("This method is not implemented yet")

    def visualize_trajectory(self):

        plot_trajectory(
            self.tool_positions,
            self.tool_orientations,
            self.timestamps,
            title="Tool Trajectory",
            is_2D=False,
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "tool_trajectory.png"
            ),
        )

        return

    def visualize_trajectory_2D(self, save_path=None):
        plot_trajectory(
            self.tool_positions,
            self.tool_orientations,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            title="Tool Trajectory",
            is_2D=True,
            save_path=save_path,
        )

        return

    def visualize_velocities(self):
        plot_velocity(
            self.tool_velocity,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            title="Tool Velocity",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "tool_velocity.png"
            ),
        )

        return

    def visualize_acceleration(self):
        plot_acceleration(
            acceleration=self.tool_acceleration,
            timestamps=self.timestamps,
            start_of_motion=self.start_of_motion,
            title="Tool Acceleration",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "tool_acceleration.png"
            ),
        )

        return


class InsideObject:
    def __init__(self, object_size, object_mass):

        self.object_size = object_size
        self.object_mass = object_mass

        # Initiializing Object Dynamics Params
        self.object_positions = []
        self.object_velocity = []
        self.object_acceleration = []

        ##Time corresponding to start of motion
        self.start_of_motion = 0  # secs
        self.end_of_motion = 0  # secs

        return

    def initialize_object_trajectory(self, timestamps, object_positions):
        self.object_positions = object_positions
        self.timestamps = timestamps
        return

    def compute_object_velocity(self):
        assert len(self.object_positions) > 0, "Object trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"

        dt = self.timestamps
        # Computing Linear Velocity
        linear_velocity_vx = np.reshape(
            np.gradient(self.object_positions[:, 0], dt), (-1, 1)
        )
        linear_velocity_vy = np.reshape(
            np.gradient(self.object_positions[:, 1], dt), (-1, 1)
        )
        linear_velocity_vz = np.reshape(
            np.gradient(self.object_positions[:, 2], dt), (-1, 1)
        )

        self.object_velocity = np.hstack(
            (linear_velocity_vx, linear_velocity_vy, linear_velocity_vz)
        )
        # [m/s]

    def compute_object_acceleration(self):
        assert len(self.object_positions) > 0, "Object trajectory is empty"
        assert len(self.timestamps) > 0, "Timestamps are empty"

        linear_acceleration = np.gradient(self.object_velocity, self.timestamps, axis=0)
        # Filtering the linear acceleration
        linear_acceleration = butter_lowpass_filter(
            linear_acceleration, cutoff=6, fs=50, order=3
        )

        self.object_acceleration = linear_acceleration
        # [m/s^2]

    def filter_object_trajectory_on_start_end_time(self):
        assert len(self.timestamps) > 0, "Timestamps are empty"
        assert self.start_of_motion != 0, "Start of motion is not set"
        assert self.end_of_motion != 0, "End of motion is not set"

        # Filtering the timestamps
        start_idx = np.argmin(np.abs(self.timestamps - self.start_of_motion))
        end_idx = np.argmin(np.abs(self.timestamps - self.end_of_motion))

        self.object_positions = self.object_positions[start_idx:end_idx]
        self.timestamps = self.timestamps[start_idx:end_idx]
        self.timestamps = self.timestamps - self.timestamps[0]

        self.object_velocity = self.object_velocity[start_idx:end_idx]
        self.object_acceleration = self.object_acceleration[start_idx:end_idx]

        self.start_of_motion = self.timestamps[0]
        self.end_of_motion = self.timestamps[-1]

        return

    def visualize_trajectory(self):

        plot_trajectory(
            self.object_positions,
            None,
            self.timestamps,
            title="Object Trajectory",
            is_2D=False,
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "object_trajectory.png"
            ),
        )

        return

    def visualize_trajectory_2D(self):
        plot_trajectory(
            self.object_positions,
            None,
            self.timestamps,
            title="Object Trajectory",
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            is_2D=True,
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "object_trajectory_2D.png"
            ),
        )

        return

    def visualize_velocities(self):
        plot_velocity(
            self.object_velocity,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            title="Object Velocity",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "object_velocity.png"
            ),
        )

        return

    def visualize_acceleration(self):
        plot_acceleration(
            self.object_acceleration,
            self.timestamps,
            start_of_motion=self.start_of_motion,
            end_of_motion=self.end_of_motion,
            title="Object Acceleration",
            save_path=os.path.join(
                os.path.dirname(MAIN_DIR), "plots", "object_acceleration.png"
            ),
        )

        return
