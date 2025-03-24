import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Slerp
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, interp1d
from collections import defaultdict
from scipy.spatial import cKDTree
import pandas as pd
from scipy.optimize import linear_sum_assignment
import collections
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
from tqdm import tqdm
import os
import pandas as pd
from matplotlib.animation import FuncAnimation

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree, ConvexHull


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_frame(ax, position, quaternion, label, axis_length=0.05):
    """Plots a 3D coordinate frame given position and quaternion"""
    # Convert quaternion to rotation matrix
    R_mat = R.from_quat(quaternion).as_matrix()

    # Define axis directions in local frame
    axes = np.eye(3) * axis_length

    # Transform axes to world frame
    world_axes = R_mat @ axes

    # Plot each axis with different color
    colors = ["r", "g", "b"]
    for i in range(3):
        ax.quiver(
            position[0],
            position[1],
            position[2],
            world_axes[0, i],
            world_axes[1, i],
            world_axes[2, i],
            color=colors[i],
            label=f"{label} {['X', 'Y', 'Z'][i]}" if i == 0 else "",
        )

    return


def plot_trajectory(
    positions,
    orientations,
    timestamps,
    start_of_motion=None,
    end_of_motion=None,
    title="Trajectory",
    is_2D=False,
    save_path=None,
    comparison_trajectory=None,
):
    """
    Plots the trajectory of a marker in 3D space.
    """

    if is_2D:
        if orientations is not None:
            fig, axs = plt.subplots(6, 1, figsize=(10, 10))
        else:
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        fig.tight_layout(pad=3.0)

        axs[0].plot(timestamps, positions[:, 0], "r-", label="Trajectory")
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("X")

        axs[1].plot(timestamps, positions[:, 1], "g-", label="Trajectory")
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("Y")

        axs[2].plot(timestamps, positions[:, 2], "b-", label="Trajectory")
        axs[2].set_xlabel("time")
        axs[2].set_ylabel("Z")

        if orientations is not None:
            rotations = R.from_quat(orientations)
            euler_angles = rotations.as_euler("xyz", degrees=True)
            axs[3].plot(timestamps, euler_angles[:, 0], "r-", label="Roll")
            axs[4].plot(timestamps, euler_angles[:, 1], "g-", label="Pitch")
            axs[5].plot(timestamps, euler_angles[:, 2], "b-", label="Yaw")

        if start_of_motion is not None:
            axs[0].axvline(x=start_of_motion, color="r", linestyle="--")
            axs[1].axvline(x=start_of_motion, color="g", linestyle="--")
            axs[2].axvline(x=start_of_motion, color="b", linestyle="--")
            if orientations is not None:
                axs[3].axvline(x=start_of_motion, color="r", linestyle="--")
                axs[4].axvline(x=start_of_motion, color="g", linestyle="--")
                axs[5].axvline(x=start_of_motion, color="b", linestyle="--")

        if end_of_motion is not None:
            axs[0].axvline(x=end_of_motion, color="r", linestyle="--")
            axs[1].axvline(x=end_of_motion, color="g", linestyle="--")
            axs[2].axvline(x=end_of_motion, color="b", linestyle="--")
            if orientations is not None:
                axs[3].axvline(x=end_of_motion, color="r", linestyle="--")
                axs[4].axvline(x=end_of_motion, color="g", linestyle="--")
                axs[5].axvline(x=end_of_motion, color="b", linestyle="--")

        fig.suptitle(title)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        return

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(
        positions[:, 0], positions[:, 1], positions[:, 2], "b-", label="Trajectory"
    )
    if comparison_trajectory is not None:
        ax.plot3D(
            comparison_trajectory[:, 0],
            comparison_trajectory[:, 1],
            comparison_trajectory[:, 2],
            "r-",
            label="Comparison Trajectory",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if orientations is not None:
        rotations = R.from_quat(orientations)
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            rotations.as_matrix()[:, 0, 0],
            rotations.as_matrix()[:, 0, 1],
            rotations.as_matrix()[:, 0, 2],
            length=0.1,
            normalize=True,
            color="r",
        )
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            rotations.as_matrix()[:, 1, 0],
            rotations.as_matrix()[:, 1, 1],
            rotations.as_matrix()[:, 1, 2],
            length=0.1,
            normalize=True,
            color="g",
        )
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            rotations.as_matrix()[:, 2, 0],
            rotations.as_matrix()[:, 2, 1],
            rotations.as_matrix()[:, 2, 2],
            length=0.1,
            normalize=True,
            color="b",
        )
        ax.legend()
        ax.set_title("3D Trajectory with Orientation")

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return


def plot_acceleration(
    acceleration,
    timestamps,
    start_of_motion=None,
    end_of_motion=None,
    title="Acceleration",
    save_path=None,
):
    """
    Plots the acceleration of a marker in 3D space.
    """
    fig, axs = plt.subplots(len(acceleration[0]), 1, figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    axs[0].plot(timestamps, acceleration[:, 0], "r-", label="X")
    axs[0].set_ylabel("Ax")
    axs[0].set_xlabel("Time")
    axs[0].legend()

    axs[1].plot(timestamps, acceleration[:, 1], "g-", label="Y")
    axs[1].set_ylabel("Ay")
    axs[1].set_xlabel("Time")
    axs[1].legend()

    axs[2].plot(timestamps, acceleration[:, 2], "b-", label="Z")
    axs[2].set_ylabel("Az")
    axs[2].set_xlabel("Time")
    axs[2].legend()

    if start_of_motion is not None:
        axs[0].axvline(x=start_of_motion, color="r", linestyle="--")
        axs[1].axvline(x=start_of_motion, color="g", linestyle="--")
        axs[2].axvline(x=start_of_motion, color="b", linestyle="--")

    if end_of_motion is not None:
        axs[0].axvline(x=end_of_motion, color="r", linestyle="--")
        axs[1].axvline(x=end_of_motion, color="g", linestyle="--")
        axs[2].axvline(x=end_of_motion, color="b", linestyle="--")

    if len(acceleration[0]) == 6:
        axs[3].plot(timestamps, acceleration[:, 3], "r-", label="X")
        axs[3].set_ylabel("Ax")
        axs[3].set_xlabel("Time")
        axs[3].legend()

        axs[4].plot(timestamps, acceleration[:, 4], "g-", label="Y")
        axs[4].set_ylabel("Ay")
        axs[4].set_xlabel("Time")
        axs[4].legend()

        axs[5].plot(timestamps, acceleration[:, 5], "b-", label="Z")
        axs[5].set_ylabel("Az")
        axs[5].set_xlabel("Time")
        axs[5].legend()

    fig.suptitle(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return


def plot_velocity(
    velocity,
    timestamps,
    start_of_motion=None,
    end_of_motion=None,
    title="Velocity",
    save_path=None,
):
    """
    Plots the velocity of a marker in 3D space.
    """
    fig, axs = plt.subplots(len(velocity[0]), 1, figsize=(10, 10))
    fig.tight_layout(pad=3.0)

    axs[0].plot(timestamps, velocity[:, 0], "r-", label="X")
    axs[0].set_ylabel("Vx")
    axs[0].set_xlabel("Time")
    axs[0].legend()

    axs[1].plot(timestamps, velocity[:, 1], "g-", label="Y")
    axs[1].set_ylabel("Vy")
    axs[1].set_xlabel("Time")
    axs[1].legend()

    axs[2].plot(timestamps, velocity[:, 2], "b-", label="Z")
    axs[2].set_ylabel("Vz")
    axs[2].set_xlabel("Time")
    axs[2].legend()

    if len(velocity[0]) == 6:
        axs[3].plot(timestamps, velocity[:, 3], "r-", label="X")
        axs[3].set_ylabel("Wx")
        axs[3].set_xlabel("Time")
        axs[3].legend()

        axs[4].plot(timestamps, velocity[:, 4], "g-", label="Y")
        axs[4].set_ylabel("Wy")
        axs[4].set_xlabel("Time")
        axs[4].legend()

        axs[5].plot(timestamps, velocity[:, 5], "b-", label="Z")
        axs[5].set_ylabel("Wz")
        axs[5].set_xlabel("Time")
        axs[5].legend()

    if start_of_motion is not None:
        axs[0].axvline(x=start_of_motion, color="r", linestyle="--")
        axs[1].axvline(x=start_of_motion, color="g", linestyle="--")
        axs[2].axvline(x=start_of_motion, color="b", linestyle="--")
        if len(velocity[0]) == 6:
            axs[3].axvline(x=start_of_motion, color="r", linestyle="--")
            axs[4].axvline(x=start_of_motion, color="g", linestyle="--")
            axs[5].axvline(x=start_of_motion, color="b", linestyle="--")

    if end_of_motion is not None:
        axs[0].axvline(x=end_of_motion, color="r", linestyle="--")
        axs[1].axvline(x=end_of_motion, color="g", linestyle="--")
        axs[2].axvline(x=end_of_motion, color="b", linestyle="--")
        if len(velocity[0]) == 6:
            axs[3].axvline(x=end_of_motion, color="r", linestyle="--")
            axs[4].axvline(x=end_of_motion, color="g", linestyle="--")
            axs[5].axvline(x=end_of_motion, color="b", linestyle="--")

    fig.suptitle(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return


def butter_lowpass_filter(data, cutoff=5, fs=50, order=3):
    """
    Apply a Butterworth low-pass filter to angular velocity data.

    Args:
        data (np.array): (N, 3) Angular velocity
        cutoff (float): Cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order

    Returns:
        np.array: Smoothed angular velocity (N, 3)
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)  # Apply filter


def process_timestamped_trajectory(
    mocap_dataframe,
    xyz_start_end_ids,
    orientation_start_end_ids=None,
    verbose=True,
    filter_window=31,
    filter_order=3,
    object=False,
    marker_trajectory=None,
):
    """
    Process the timestamped trajectory data from the mocap dataframe.
    Args:
        mocap_dataframe (pd.DataFrame): Mocap dataframe
        xyz_start_end_ids (list): Start and end ids for the xyz positions
        orientation_start_end_ids (list): Start and end ids for the orientation
        verbose (bool): Whether to print logs or not
        filter_window (int): Window size for the Savitzky-Golay filter
        filter_order (int): Order of the Savitzky-Golay filter
    Returns:
        tuple: Processed timestamps, positions, orientations
    """

    mocap_dataframe_keys = mocap_dataframe.keys()

    timestamps = mocap_dataframe[mocap_dataframe_keys[1]].to_numpy()

    ##Interpolating the missing values in the trajectory
    positions = mocap_dataframe[
        mocap_dataframe_keys[xyz_start_end_ids[0] : xyz_start_end_ids[1]]
    ].to_numpy()
    ##Checking if the object disappeared at some point
    og_traj = positions.copy()
    # if object:
    #     # logger.debug(f"Positions: {positions[1000:1300]}")
    #     positions[1200:1300] = positions[1199]
    #     constant_mask = np.all(positions == np.roll(positions, shift=1, axis=0), axis=1)
    #     # positions[constant_mask, 0] = 0
    #     # positions[constant_mask, 1] = 0
    #     # positions[constant_mask, 2] = 0
    #     positions[constant_mask] = np.nan

    ##NOTE: This is a test code for cases when object will have missing timestamps
    # mocap_dataframe[mocap_dataframe_keys[xyz_start_end_ids[0]]][1200:1300] = positions[
    #     1199
    # ][0]
    # mocap_dataframe[mocap_dataframe_keys[xyz_start_end_ids[0] + 1]][1200:1300] = (
    #     positions[1199][1]
    # )
    # mocap_dataframe[mocap_dataframe_keys[xyz_start_end_ids[0] + 2]][1200:1300] = (
    #     positions[1199][2]
    # )

    if np.isnan(positions).any():
        if verbose:
            logger.info("Interpolating the missing values in the trajectory.")
        if marker_trajectory is None:
            positions = (
                mocap_dataframe[
                    mocap_dataframe_keys[xyz_start_end_ids[0] : xyz_start_end_ids[1]]
                ]
                .interpolate(method="from_derivatives")
                .to_numpy()
            )

        else:
            interp_funcs = {}
            ##NOTE (OMEY): Verify that this can be done with marker position
            valid_indices = np.where(~np.isnan(positions).any(axis=1))[0]
            for id, axis in enumerate(["x_dep", "y_dep", "z_dep"]):
                interp_funcs[axis] = interp1d(
                    marker_trajectory[valid_indices, id],
                    positions[valid_indices, id],
                    kind="nearest",
                    fill_value="extrapolate",
                )

            for id, axis in enumerate(["x_dep", "y_dep", "z_dep"]):
                missing_mask = np.where(np.isnan(positions).any(axis=1))[0]
                positions[missing_mask, id] = interp_funcs[axis](
                    marker_trajectory[missing_mask, id]
                )

    ##Filterning the positions
    positions[:, 0] = (
        savgol_filter(positions[:, 0], filter_window, filter_order) / 1000
    )  # Converting to meters
    positions[:, 1] = savgol_filter(positions[:, 1], filter_window, filter_order) / 1000
    positions[:, 2] = savgol_filter(positions[:, 2], filter_window, filter_order) / 1000

    ##Interpolating the missing values in the orientation
    orientations = []
    if orientation_start_end_ids != None:
        orientations = mocap_dataframe[
            mocap_dataframe_keys[
                orientation_start_end_ids[0] : orientation_start_end_ids[1]
            ]
        ].to_numpy()

        if verbose:
            logger.info("Interpolating the missing values in the orientation.")
        valid_indices = np.where(~np.isnan(orientations).any(axis=1))[0]

        valid_orientations = R.from_quat(orientations[valid_indices])
        slerp_outputs = Slerp(
            timestamps[valid_indices],
            valid_orientations,
        )

        if np.isnan(orientations).any():
            nan_indices = np.where(np.isnan(orientations).any(axis=1))[0]
            orientations[nan_indices] = slerp_outputs(timestamps[nan_indices]).as_quat()

    return positions, orientations


def interpolate_using_neighbors(trajectories, neighbor_count=3):
    """
    Interpolates missing marker positions using the velocity of neighboring markers.

    Args:
        trajectories (dict): {marker_id: [(t, x, y, z), ...]} sorted by time.
        neighbor_count (int): Number of nearest neighbors to use for velocity estimation.

    Returns:
        dict: Interpolated trajectories {marker_id: [(t, x, y, z), ...]}.
    """
    interpolated_trajectories = {}
    all_timestamps = sorted(
        set(t for traj in trajectories.values() for t, _, _, _ in traj)
    )

    # Convert trajectories to a structured format
    marker_positions = {
        marker: {t: np.array([x, y, z]) for t, x, y, z in traj}
        for marker, traj in trajectories.items()
    }

    for marker_id, traj in trajectories.items():
        traj = sorted(traj, key=lambda x: x[0])  # Ensure sorted by time
        timestamps = np.array([p[0] for p in traj])
        positions = np.array([p[1:] for p in traj])

        if len(timestamps) < 2:
            interpolated_trajectories[marker_id] = traj
            continue

        interpolated_positions = []

        for id, t in enumerate(all_timestamps):
            if (
                t in marker_positions[marker_id]
            ):  # If timestamp exists, use actual value
                interpolated_positions.append((t, *marker_positions[marker_id][t]))
            else:
                # Find neighboring markers that have data at time t
                valid_neighbors = []

                for other_marker, other_traj in marker_positions.items():
                    if other_marker != marker_id and t in other_traj:
                        valid_neighbors.append(other_traj[t])

                if (
                    len(valid_neighbors) < 2
                ):  # Not enough data for reliable interpolation
                    continue

                # Use KDTree to find nearest neighbors
                if id >= len(positions):
                    kd_tree = cKDTree(valid_neighbors)
                    _, neighbor_indices = kd_tree.query(
                        positions[-1], k=min(neighbor_count, len(valid_neighbors))
                    )
                else:
                    kd_tree = cKDTree(valid_neighbors)
                    _, neighbor_indices = kd_tree.query(
                        positions[id], k=min(neighbor_count, len(valid_neighbors))
                    )

                # Compute average velocity of nearest neighbors
                avg_velocity = np.zeros(3)
                count = 0
                for idx in neighbor_indices:
                    neighbor_id = list(marker_positions.keys())[idx]

                    # Get the closest two timestamps surrounding 't'
                    neighbor_timestamps = sorted(marker_positions[neighbor_id].keys())
                    prev_t = max(
                        [ts for ts in neighbor_timestamps if ts < t], default=None
                    )
                    next_t = min(
                        [ts for ts in neighbor_timestamps if ts > t], default=None
                    )

                    # If no valid surrounding timestamps, continue
                    if prev_t is None or next_t is None:
                        continue

                    # Interpolate positions at prev_t and next_t
                    prev_pos = marker_positions[neighbor_id][prev_t]
                    next_pos = marker_positions[neighbor_id][next_t]

                    # Linear interpolation for position at 't'
                    interp_pos = prev_pos + (next_pos - prev_pos) * (t - prev_t) / (
                        next_t - prev_t
                    )

                    # Compute velocity: (position at next_t - position at prev_t) / (next_t - prev_t)
                    velocity = (next_pos - prev_pos) / (next_t - prev_t)

                    avg_velocity += velocity
                    count += 1

                if count > 0:
                    avg_velocity /= count
                    predicted_position = positions[-1] + avg_velocity * (
                        t - timestamps[-1]
                    )
                    interpolated_positions.append((t, *predicted_position))

        interpolated_trajectories[marker_id] = interpolated_positions

    return interpolated_trajectories


def find_missing_centroid(marker_positions):
    # Compute the convex hull of the known markers
    hull = ConvexHull(marker_positions)

    # Find the boundary points
    boundary_indices = np.unique(hull.simplices)
    boundary_points = marker_positions[boundary_indices]

    # Compute the centroid of the boundary points
    centroid = np.mean(boundary_points, axis=0)
    return centroid


def array_to_SE3(pose_array: np.ndarray):

    mat = np.eye(4)

    position = pose_array[:3]
    rotation_mat = R.from_quat(pose_array[3:]).as_matrix()

    mat[:3, :3] = rotation_mat
    mat[:3, 3] = position

    return mat


def SE3_to_array(SE3_Mat):

    pose_array = np.zeros(7)

    pose_array[:3] = SE3_Mat[:3, 3]
    pose_array[3:] = R.from_matrix(SE3_Mat[:3, :3]).as_quat()

    return pose_array


def visualize_markers(positions, frames: list = None):
    """
    Visualizes the 3D positions of markers.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )

    if frames:
        for frame in frames:
            plot_frame(
                ax=ax, position=frame[1][:3], quaternion=frame[1][3:], label=frame[0]
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def visualize_marker_motion(timestamps, markers, saving_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()  # Clear previous frame
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

        timestamp = timestamps[frame]

        # Plot markers
        ax.scatter(
            markers[timestamp][:, 0],
            markers[timestamp][:, 1],
            markers[timestamp][:, 2],
            color="b",
            label="Markers",
        )

        ax.legend()

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=len(timestamps), interval=100
    )  # Adjust interval as needed
    ani.save(os.path.join(os.path.dirname(MAIN_DIR), "plots", saving_filename))
    # Show animation
    plt.show()


def filter_spurious_markers(markers, eps=0.05, min_samples=3):
    """Remove spurious markers using DBSCAN clustering."""
    if len(markers) < 3:
        return markers  # Not enough points for clustering

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(markers)

    # Keep only markers belonging to the largest cluster
    core_mask = labels != -1
    return markers[core_mask]


def farthest_point_sampling(points, k):
    """Select k points using farthest point sampling (FPS)."""
    if len(points) <= k:
        return points  # No need for sampling

    selected = [points[np.random.randint(len(points))]]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(points[:, None, :] - np.array(selected)[None, :, :], axis=2),
            axis=1,
        )
        next_point = points[np.argmax(dists)]
        selected.append(next_point)

    return np.array(selected)


def interpolate_missing_markers(markers, k=42):
    """If there are fewer than k markers, interpolate using nearest neighbors."""
    if len(markers) >= k:
        return markers

    tree = KDTree(markers)
    additional_points = []

    while len(markers) + len(additional_points) < k:
        query_point = np.mean(markers, axis=0) + 0.01 * np.random.randn(
            3
        )  # Perturbation
        _, idx = tree.query(query_point)
        additional_points.append(markers[idx])

    return np.vstack((markers, additional_points[: k - len(markers)]))


def pca_sort(markers):
    """Sort markers based on PCA principal axis for spatial consistency."""
    pca = PCA(n_components=3)
    markers_pca = pca.fit_transform(markers)

    sorted_indices = np.lexsort(
        (markers_pca[:, 1], markers_pca[:, 0])
    )  # Sort by first two PCA axes
    return markers[sorted_indices]


def process_markers(detected_markers, eps, target_count=42):
    """
    Given a set of detected markers (some missing/spurious), return exactly 42 markers.
    """
    detected_markers = np.array(detected_markers)
    # Step 1: Remove spurious markers
    filtered_markers = filter_spurious_markers(detected_markers, eps=eps, min_samples=3)
    # Step 2: Ensure exactly 42 markers
    if len(filtered_markers) > target_count:
        filtered_markers = farthest_point_sampling(filtered_markers, target_count)

    # Step 3: Sort markers consistently using PCA
    sorted_markers = pca_sort(filtered_markers)

    return sorted_markers


def process_marker_timestamped_trajectory_no_id(
    mocap_dataframe: pd.DataFrame,
    package_dims: list | np.ndarray,
    xyz_start_end_ids: list,
    num_markers: int,
    start_of_motion,
    end_of_motion,
    saving_fileheader: str = None,
):
    marker_dict = {}
    timestamps = mocap_dataframe[mocap_dataframe.keys()[1]]
    start_of_motion_id = np.argmin(
        np.abs(timestamps - start_of_motion)
    )  # Find the index of the closest timestamp to start_of_motion
    end_of_motion_id = np.argmin(
        np.abs(timestamps - end_of_motion)
    )  # Find the index of the closest timestamp to end_of_motion

    initial_marker_position = []

    for id, row in mocap_dataframe.iterrows():
        marker_data = row[xyz_start_end_ids[0] : xyz_start_end_ids[1]]
        if id == 0:
            for col in marker_data.index:
                if col.startswith("X") and not np.isnan(marker_data[col]):
                    initial_marker_position.append(
                        (
                            marker_data[col],
                            marker_data[col.replace("X", "Y")],
                            marker_data[col.replace("X", "Z")],
                        )
                    )

        if id < start_of_motion_id or id > end_of_motion_id:
            continue
        ##Add a filter to remove markers beyond a certain distance
        current_positions = {}
        col_counts = 1
        current_marker_positions = []
        for col in marker_data.index:
            if col.startswith("X") and not np.isnan(marker_data[col]):
                current_positions[col_counts] = (
                    marker_data[col],
                    marker_data[col.replace("X", "Y")],
                    marker_data[col.replace("X", "Z")],
                )
                col_counts += 1
                current_marker_positions.append(
                    (
                        marker_data[col],
                        marker_data[col.replace("X", "Y")],
                        marker_data[col.replace("X", "Z")],
                    )
                )

        current_marker_positions = process_markers(
            np.array(current_marker_positions).reshape(-1, 3),
            target_count=num_markers,
            eps=max(package_dims) * 1000 / 2,
        )
        timestamp = row[mocap_dataframe.keys()[1]] - timestamps[start_of_motion_id]
        marker_dict[timestamp] = current_marker_positions / 1000

        # if id >= 1000:
        #     print(len(current_marker_positions))
        #     visualize_markers(marker_dict[timestamp])

    saving_fileheader = f"{saving_fileheader}_markers.gif"
    # visualize_marker_motion(
    #     timestamps=list(marker_dict.keys()),
    #     markers=marker_dict,
    #     saving_filename=saving_fileheader,
    # )
    initial_marker_position = np.array(initial_marker_position).reshape(-1, 3) / 1000
    return initial_marker_position, marker_dict


def process_marker_timestamped_trajectory(
    mocap_dataframe: pd.DataFrame,
    xyz_start_end_ids: list,
    num_markers: int,
    filter_window=51,
    filter_order=3,
    plot=False,
    saving_fileheader: str = None,
):
    trajectories = defaultdict(list)
    global_marker_ids = 0
    distance_threshold = 0.01 * 1000  ##NOTE: Need to fine-tune this
    global_marker_ids = {}
    next_global_id = 1
    inactive_markers = collections.deque(maxlen=num_markers)
    previous_positions = {}

    for id, row in mocap_dataframe.iterrows():
        timestamp = row[mocap_dataframe.keys()[1]]
        marker_data = row[xyz_start_end_ids[0] : xyz_start_end_ids[1]]

        ##Add a filter to remove markers beyond a certain distance
        current_positions = {}
        col_counts = 1
        for col in marker_data.index:
            if col.startswith("X") and not np.isnan(marker_data[col]):
                current_positions[col_counts] = (
                    marker_data[col],
                    marker_data[col.replace("X", "Y")],
                    marker_data[col.replace("X", "Z")],
                )
                col_counts += 1

        if not global_marker_ids:
            # First frame: Assign new global IDs to all markers
            for marker_id, pos in current_positions.items():
                global_marker_ids[marker_id] = next_global_id
                previous_positions[next_global_id] = pos
                trajectories[next_global_id].append((timestamp, *pos))
                next_global_id += 1
        else:
            # Associate current markers with previous markers
            association = associate_markers(
                previous_positions, current_positions, distance_threshold
            )

            # Track missing markers
            disappeared = set(global_marker_ids.keys()) - set(association.values())
            for missing in disappeared:
                inactive_markers.append(global_marker_ids[missing])  # Mark as inactive

            # Update global marker tracking
            new_global_ids = {}
            for curr_id, prev_id in association.items():
                new_global_ids[curr_id] = prev_id
                previous_positions[prev_id] = current_positions[curr_id]
                trajectories[prev_id].append((timestamp, *current_positions[curr_id]))

            # Assign new IDs to unassociated markers
            for curr_id in current_positions.keys():
                if curr_id not in new_global_ids:
                    if inactive_markers:
                        # Reuse old ID
                        reused_id = inactive_markers.popleft()
                        new_global_ids[curr_id] = reused_id
                    elif next_global_id <= num_markers:
                        # Assign a new global ID (only if within 30 limit)
                        new_global_ids[curr_id] = next_global_id
                        next_global_id += 1
                    else:
                        # Ignore extra markers beyond 30
                        continue

                    trajectories[new_global_ids[curr_id]].append(
                        (timestamp, *current_positions[curr_id])
                    )

            global_marker_ids = new_global_ids  # Update global IDs for next frame

    trajectories = interpolate_marker_trajectories(trajectories)

    filtered_trajectories = {}
    # Plotting the   trajectories

    for marker_id, trajectory in trajectories.items():
        marker_1_trajectory = np.array(trajectory)
        # marker_trajectory_x = marker_1_trajectory[:, 1]/1000
        # marker_trajectory_y = marker_1_trajectory[:, 2]/1000
        # marker_trajectory_z = marker_1_trajectory[:, 3]/1000
        data_length = len(marker_1_trajectory)
        window_length = filter_window
        if window_length > data_length:
            window_length = data_length if data_length % 2 == 1 else data_length - 1
            if window_length < 3:  # Minimum valid window size is 3
                logger.info("Data too short for filtering.")
                continue  # Return original data if filtering is not possible

        current_order = filter_order
        if current_order >= window_length:
            current_order = window_length - 1
        kernel_size = min(
            5,
            (
                len(marker_1_trajectory)
                if len(marker_1_trajectory) % 2 == 1
                else len(marker_1_trajectory) - 1
            ),
        )
        marker_trajectory_x = medfilt(
            savgol_filter(marker_1_trajectory[:, 1], window_length, current_order)
            / 1000,
            kernel_size=kernel_size,
        )
        marker_trajectory_y = medfilt(
            savgol_filter(marker_1_trajectory[:, 2], window_length, current_order)
            / 1000,
            kernel_size=kernel_size,
        )
        marker_trajectory_z = medfilt(
            savgol_filter(marker_1_trajectory[:, 3], window_length, current_order)
            / 1000,
            kernel_size=kernel_size,
        )

        filtered_trajectories[marker_id] = np.array(
            [
                marker_1_trajectory[:, 0],
                marker_trajectory_x,
                marker_trajectory_y,
                marker_trajectory_z,
            ]
        ).T

    logger.info(f"Found {len(filtered_trajectories)} markers in the trajectory.")

    if plot or saving_fileheader is not None:
        image_filename = None
        if saving_fileheader is not None:
            image_filename = f"{saving_fileheader}_{marker_id}.png"
        for marker_id, filtered_trajectory in filtered_trajectories.items():
            marker_trajectory_x = filtered_trajectory[:, 1]
            marker_trajectory_y = filtered_trajectory[:, 2]
            marker_trajectory_z = filtered_trajectory[:, 3]
            plot_trajectory(
                positions=np.array(
                    [marker_trajectory_x, marker_trajectory_y, marker_trajectory_z]
                ).T,
                orientations=None,
                timestamps=filtered_trajectory[:, 0],
                title=f"Marker Trajectory {marker_id}, {mocap_dataframe.keys()[xyz_start_end_ids[0]],mocap_dataframe.keys()[xyz_start_end_ids[0]+1],mocap_dataframe.keys()[xyz_start_end_ids[0]+2]}",
                is_2D=True,
                save_path=image_filename,
            )

    return filtered_trajectories


def compute_smooth_velocity(positions, timestamps, window=5, poly=2):
    """Compute velocity and apply Savitzky-Golay smoothing."""
    velocities = np.gradient(positions, timestamps)

    # Apply smoothing only if there are enough points
    if len(velocities) >= window:
        velocities = savgol_filter(velocities, window, poly, mode="nearest")

    return velocities


def filter_marker_start_end(trajectories, start_time, end_time):
    """Filter trajectories based on start and end time."""
    filtered_trajectories = {}

    for marker_id, traj in trajectories.items():
        filtered_traj = [
            (t, x, y, z) for t, x, y, z in traj if start_time <= t <= end_time
        ]
        if len(filtered_traj) > 0:
            filtered_trajectories[marker_id] = filtered_traj

    return filtered_trajectories


def interpolate_marker_trajectories(trajectories):
    interpolated_trajectories = {}

    all_timestamps = sorted(
        set(t for traj in trajectories.values() for t, _, _, _ in traj)
    )

    for marker_id, traj in trajectories.items():
        if len(traj) < 2:
            # Skip markers with only one data point
            interpolated_trajectories[marker_id] = traj
            continue

        # Sort by timestamp
        traj = sorted(traj, key=lambda x: x[0])
        timestamps, xs, ys, zs = zip(*traj)

        # Compute first-order derivatives (velocities)
        vx = np.gradient(xs, timestamps)
        vy = np.gradient(ys, timestamps)
        vz = np.gradient(zs, timestamps)

        velocities = np.vstack((vx, vy, vz)).T

        # # Apply smoothing only if there are enough points
        window = 5
        poly = 3
        if len(velocities) >= window:
            velocities = savgol_filter(velocities, window, poly, mode="nearest")

        vx = velocities[:, 0]
        vy = velocities[:, 1]
        vz = velocities[:, 2]

        # Interpolating missing values using velocity
        interpolated_traj = []
        last_x, last_y, last_z = xs[0], ys[0], zs[0]
        last_vx, last_vy, last_vz = vx[0], vy[0], vz[0]
        last_t = timestamps[0]

        for t in all_timestamps:
            if t in timestamps:
                # Update known values
                idx = np.where(timestamps == t)[0][0]
                last_x, last_y, last_z = xs[idx], ys[idx], zs[idx]
                last_vx, last_vy, last_vz = vx[idx], vy[idx], vz[idx]
                interpolated_traj.append((t, last_x, last_y, last_z))
            # else:
            #     # Predict position using velocity
            #     dt = t - last_t

            #     last_x += last_vx * dt
            #     last_y += last_vy * dt
            #     last_z += last_vz * dt

            # last_t = t  # Update last known timestamp

        interpolated_trajectories[marker_id] = interpolated_traj

    return interpolated_trajectories


def interpolate_using_group_velocity(trajectories):
    interpolated_trajectories = {}

    for marker_id, traj in trajectories.items():
        if len(traj) < 2:
            interpolated_trajectories[marker_id] = traj
            continue

        # Sort by timestamp
        traj = sorted(traj, key=lambda x: x[0])
        timestamps = np.array([p[0] for p in traj])
        positions = np.array([p[1:] for p in traj])  # (x, y, z)

        # Compute velocity (first-order difference)
        velocities = np.gradient(positions, timestamps, axis=0)

        # Store marker trajectory
        interpolated_trajectories[marker_id] = {
            t: (p, v) for t, p, v in zip(timestamps, positions, velocities)
        }

    # Find the global timestamp range
    full_timestamps = sorted(
        set(t for traj in trajectories.values() for t, _, _, _ in traj)
    )

    # Interpolating missing positions
    for marker_id, marker_traj in interpolated_trajectories.items():
        known_timestamps = list(marker_traj.keys())

        last_known_pos, last_known_vel = None, None
        for t in full_timestamps:
            if t in known_timestamps:
                # Update last known position & velocity
                last_known_pos, last_known_vel = marker_traj[t]
            else:
                # Estimate velocity from well-tracked markers
                neighbor_velocities = []
                for other_id, other_traj in interpolated_trajectories.items():
                    if other_id == marker_id:
                        continue

                    if t in other_traj:
                        neighbor_velocities.append(other_traj[t][1])

                if neighbor_velocities:
                    avg_velocity = np.mean(neighbor_velocities, axis=0)
                else:
                    avg_velocity = last_known_vel  # Use last known velocity

                # Predict missing position
                if last_known_pos is not None and avg_velocity is not None:
                    estimated_pos = last_known_pos + avg_velocity
                    marker_traj[t] = (estimated_pos, avg_velocity)

        # Store final trajectory
        interpolated_trajectories[marker_id] = {
            t: (p, v) for t, (p, v) in marker_traj.items()
        }
    # Convert to interpolated_trajectories
    for marker_id, marker_traj in interpolated_trajectories.items():
        interpolated_trajectories[marker_id] = [
            (t, *p) for t, (p, _) in sorted(marker_traj.items())
        ]
    return interpolated_trajectories


def associate_markers(prev_positions, current_positions, distance_threshold):
    """
    Associates markers between two frames using nearest-neighbor matching.
    Uses Hungarian algorithm to optimally pair markers.
    """
    if not prev_positions or not current_positions:
        return {}

    prev_ids, prev_coords = zip(*prev_positions.items()) if prev_positions else ([], [])
    curr_ids, curr_coords = (
        zip(*current_positions.items()) if current_positions else ([], [])
    )

    prev_coords = np.array(prev_coords)
    curr_coords = np.array(curr_coords)

    # Compute pairwise distances
    # distance_matrix = np.linalg.norm(
    #     prev_coords[:, None] - curr_coords[None, :], axis=2
    # )
    distance_matrix = cdist(prev_coords, curr_coords)
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Map previous marker IDs to new marker IDs
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if distance_matrix[r, c] < distance_threshold:  # Threshold for reassociation
            mapping[curr_ids[c]] = prev_ids[r]

    return mapping


def visualize_no_id_package_motion(package: dict, tool, object):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set axis limits
    # Using package motion limits
    # Set aspect ratio
    # ax.set_box_aspect([1, 1, 1])
    all_positions = []
    for marker in package.values():
        all_positions.append(marker)

    all_positions = np.vstack(all_positions)
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

    def update(frame):
        ax.clear()  # Clear previous frame
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        # Set fixed axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        timestamp = list(package.keys())[frame]
        # Plot markers
        ax.scatter(
            package[timestamp][:, 0],
            package[timestamp][:, 1],
            package[timestamp][:, 2],
            color="b",
            label="Markers",
        )
        # Plot tool
        plot_frame(
            ax=ax,
            position=tool.tool_positions[frame],
            quaternion=tool.tool_orientations[frame],
            label="Tool",
        )

        # Plot object
        plot_frame(
            ax=ax,
            position=object.object_positions[frame],
            quaternion=np.array([0, 0, 0, 1]),
            label="Object",
        )

        # Plot World
        plot_frame(
            ax=ax,
            position=np.array([0, 0, 0]),
            quaternion=np.array([0, 0, 0, 1]),
            label="World",
        )

        ax.legend()

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=len(package) - 1, interval=100
    )  # Adjust interval as needed
    ani.save(os.path.join(os.path.dirname(MAIN_DIR), "plots", "no_id_package.gif"))
    # Show animation
    plt.show()
    return
