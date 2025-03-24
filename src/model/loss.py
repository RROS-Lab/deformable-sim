import warp as wp
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import torch


@wp.kernel
def loss_sum(
    loss_val: wp.array(dtype=wp.float32), loss_sum: wp.array(dtype=wp.float32)
):
    i = wp.tid()
    wp.atomic_add(loss_sum, 0, loss_val[i])


@wp.kernel
def add_arrays(
    a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)
):
    i = wp.tid()  # Thread index
    out[i] = a[i] + b[i]  # Element-wise addition


@wp.kernel
def chamfer_loss_warp(
    points_a: wp.array(dtype=wp.vec3),
    points_b: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):

    a_index = wp.tid()

    min_dist_sim = float(1e6)

    for i in range(points_b.shape[0]):
        # dist = wp.length_sq(diff)
        dx = points_b[i][0] - points_a[a_index][0]
        dy = points_b[i][1] - points_a[a_index][1]
        dz = points_b[i][2] - points_a[a_index][2]
        dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
        min_dist_sim = wp.min(min_dist_sim, dist)

    loss[a_index] = min_dist_sim


@wp.kernel
def divide_array(arr: wp.array(dtype=float), scalar: float, out: wp.array(dtype=float)):
    i = wp.tid()  # Thread index
    out[i] = arr[i] / scalar  # Element-wise division


def chamfer_distance_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chamfer Distance between two point clouds A and B using PyTorch.

    Parameters:
        A (torch.Tensor): Nx3 tensor of 3D points.
        B (torch.Tensor): Mx3 tensor of 3D points.

    Returns:
        torch.Tensor: Chamfer Distance (scalar).
    """
    # Compute pairwise squared Euclidean distances (NxM)
    dists = torch.cdist(A, B, p=2)  # (N, M)
    # Get minimum distance from A to B (for each point in A, find closest in B)
    min_dist_A_to_B = torch.min(dists, dim=1)[0]  # (N,)

    # Get minimum distance from B to A (for each point in B, find closest in A)
    min_dist_B_to_A = torch.min(dists, dim=0)[0]  # (M,)

    # Compute Chamfer distance as sum of both losses
    chamfer_dist = (torch.mean(min_dist_A_to_B) + torch.mean(min_dist_B_to_A)) / 2

    return chamfer_dist


# Wrapper function to compute full Chamfer loss
def compute_chamfer_loss_warp(points_A: wp.array, points_B: wp.array):
    N = points_A.shape[0]
    M = points_B.shape[0]

    loss_A = wp.zeros(N, dtype=float, requires_grad=True)
    loss_B = wp.zeros(M, dtype=float, requires_grad=True)

    wp.launch(chamfer_loss_warp, dim=N, inputs=[points_A, points_B, loss_A])
    wp.launch(chamfer_loss_warp, dim=M, inputs=[points_B, points_A, loss_B])

    # Sum both losses
    loss_A_sum = wp.zeros(1, dtype=float, requires_grad=True)
    loss_B_sum = wp.zeros(1, dtype=float, requires_grad=True)
    total_loss = wp.zeros(1, dtype=float, requires_grad=True)

    wp.launch(loss_sum, dim=loss_A.shape[0], inputs=[loss_A, loss_A_sum])
    wp.launch(loss_sum, dim=loss_B.shape[0], inputs=[loss_B, loss_B_sum])
    loss_A_red = wp.zeros(1, dtype=float, requires_grad=True)
    loss_B_red = wp.zeros(1, dtype=float, requires_grad=True)

    wp.launch(
        divide_array,
        dim=loss_A_sum.shape[0],
        inputs=[loss_A_sum, 2 * N, loss_A_red],
    )
    wp.launch(
        divide_array,
        dim=loss_B_sum.shape[0],
        inputs=[loss_B_sum, 2 * M, loss_B_red],
    )

    wp.launch(
        add_arrays, dim=loss_A_red.shape[0], inputs=[loss_A_red, loss_B_red, total_loss]
    )

    return loss_A_red, loss_B_red, total_loss


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default 'l2'
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist


def main():
    wp.init()
    # Allocate a Warp array for N x 3 random points
    pointsA = wp.array(np.random.randn(100, 3), dtype=wp.vec3, requires_grad=True)
    pointsB = wp.array(np.random.randn(200, 3), dtype=wp.vec3, requires_grad=True)

    warp_chamfer_loss = compute_chamfer_loss_warp(points_A=pointsA, points_B=pointsB)
    torch_chamfer_loss = chamfer_distance_torch(
        wp.to_torch(pointsA), wp.to_torch(pointsB)
    )

    print(f"Chamfer loss computed from Torch: {torch_chamfer_loss}")
    print(f"Chamfer loss computed from Warp: {warp_chamfer_loss[2]}")

    return


if __name__ == "__main__":
    main()
