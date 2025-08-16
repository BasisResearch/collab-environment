"""Check functions related to adjacency matrices"""

import numpy as np
import torch

from collab_env.gnn.gnn import build_edge_index

def test_build_edge_index():
    positions1 = (
        np.hstack(
            [np.random.randn(10).reshape((-1, 1)), np.random.randn(10).reshape((-1, 1))]
        )
        + 5
    )  # generate 10 points from a 2D Gaussian

    positions2 = (
        np.hstack(
            [np.random.randn(10).reshape((-1, 1)), np.random.randn(10).reshape((-1, 1))]
        )
        - 5
    )  # generate 10 points from a 2D Gaussian

    positions = np.vstack((positions1, positions2))
    edge_index = build_edge_index(torch.tensor(positions[np.newaxis, :]), 1)

    for i in range(
        10
    ):  # for points in the first cluster, we shall only find edges connecting to other points to the first cluster.
        assert torch.all(edge_index[1, edge_index[0, :] == i] <= 10)

    dist = torch.cdist(
        torch.tensor(positions), torch.tensor(positions), p=2
    )  # [B*N, B*N]

    thresholds = [0.05, 0.1, 0.5, 1]

    for threshold in thresholds:
        edge_index = build_edge_index(torch.tensor(positions[np.newaxis, :]), threshold)

        for i in range(
            dist.shape[0]
        ):  # for points in the first cluster, we shall only find edges connecting to other points to the first cluster.
            j = edge_index[1, edge_index[0, :] == i]
            if len(j) == 0:
                continue
            assert torch.all(dist[i, j] <= threshold)

    # TO DO: add more tests for other functions
