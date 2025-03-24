import numpy as np

from scipy import interpolate


def interpolate_tool_data(
    timestamps,
    tool_positions,
    tool_orientations,
    tool_velocities=None,
    factor=512,
    rest_period=0.0,
):
    """
    Interpolate tool positions, orientations, and optionally velocities to have 'factor' times more samples,
    with an option to prepend a rest period where data stays at the initial values.
    Returns results as a numpy array containing wp.transform objects.

    Parameters:
    timestamps (np.ndarray): Original timestamps, shape (n,)
    tool_positions (np.ndarray): Original positions, shape (n, 3)
    tool_orientations (np.ndarray): Original orientations as quaternions, shape (n, 4)
        with format [x, y, z, w]
    tool_velocities (np.ndarray, optional): Original velocities, shape (n, 3)
    factor (int): Interpolation factor, default 512
    rest_period (float): Duration to prepend where data remains at initial values

    Returns:
    tuple: (new_timestamps, transforms, new_velocities)
        - new_timestamps (np.ndarray): Interpolated timestamps including rest period
        - transforms (np.ndarray): Numpy array of wp.transform objects containing position and orientation
        - new_velocities (np.ndarray or None): Interpolated velocities if tool_velocities was provided
    """
    n = timestamps.shape[0]

    # Get initial position and orientation
    initial_position = tool_positions[0].copy()
    initial_orientation = tool_orientations[0].copy()

    # Create interpolated timestamps for the original data range
    t_min, t_max = timestamps[0], timestamps[-1]
    interpolated_timestamps = np.linspace(t_min, t_max, n * factor)

    # Calculate number of samples for rest period based on the interpolated frequency
    if rest_period > 0:
        # Calculate time step in the interpolated data
        dt = (t_max - t_min) / (n * factor - 1)

        # Calculate number of samples needed for rest period
        rest_samples = int(rest_period / dt)

        # Create timestamps for rest period
        rest_timestamps = np.linspace(t_min - rest_period, t_min - dt, rest_samples)

        # Combine rest period timestamps with interpolated timestamps
        new_timestamps = np.concatenate([rest_timestamps, interpolated_timestamps])

        # Initialize new data arrays
        new_positions = np.zeros((len(new_timestamps), 3))
        new_orientations = np.zeros((len(new_timestamps), 4))

        # Fill rest period with initial values
        new_positions[:rest_samples] = initial_position
        new_orientations[:rest_samples] = initial_orientation

        # Interpolation will be applied only to the original time range
        start_idx = rest_samples
    else:
        new_timestamps = interpolated_timestamps
        new_positions = np.zeros((len(new_timestamps), 3))
        new_orientations = np.zeros((len(new_timestamps), 4))
        start_idx = 0

    # Create interpolation functions for each position dimension (x, y, z)
    for dim in range(3):
        f = interpolate.interp1d(
            timestamps,
            tool_positions[:, dim],
            kind="cubic",  # Cubic spline interpolation
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Apply interpolation function to get new position values
        new_positions[start_idx:, dim] = f(interpolated_timestamps)

    # Normalize quaternions to ensure unit quaternions
    norm = np.linalg.norm(tool_orientations, axis=1, keepdims=True)
    normalized_orientations = tool_orientations / norm

    # Interpolate each quaternion component separately
    for dim in range(4):
        f = interpolate.interp1d(
            timestamps,
            normalized_orientations[:, dim],
            kind="linear",  # Linear is often better for quaternions
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Apply interpolation function to get new orientation values
        new_orientations[start_idx:, dim] = f(interpolated_timestamps)

    # Renormalize the interpolated quaternions
    norm = np.linalg.norm(new_orientations, axis=1, keepdims=True)
    new_orientations = new_orientations / norm

    # Handle velocity interpolation if provided
    if tool_velocities is not None:
        initial_velocity = tool_velocities[0].copy()
        new_velocities = np.zeros((len(new_timestamps), 3))

        # Fill rest period with initial velocity (zero)
        new_velocities[:start_idx] = 0.0  # Zero velocity during rest

        # Interpolate each velocity component
        for dim in range(3):
            f = interpolate.interp1d(
                timestamps,
                tool_velocities[:, dim],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Apply interpolation function to get new velocity values
            new_velocities[start_idx:, dim] = f(interpolated_timestamps)
    else:
        new_velocities = None

    return new_timestamps, new_positions, new_orientations, new_velocities
